from threading import Lock
from typing import Dict, List, Optional, Union

import torch
from dataclasses import dataclass
from transformers import IdeficsForVisionText2Text, AutoProcessor, IdeficsProcessor

from helm.common.cache import CacheConfig
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.common.request import wrap_request_time
from helm.proxy.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.proxy.tokenizers.tokenizer import Tokenizer

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])


@dataclass(frozen=True)
class LoadedIDEFICSModelProcessor:
    """Loaded model and processor for IDEFICS."""

    model: IdeficsForVisionText2Text
    processor: IdeficsProcessor


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedIDEFICSModelProcessor]] = {
    "HuggingFaceM4/idefics-9b": None,
    "HuggingFaceM4/idefics-9b-instruct": None,
    "HuggingFaceM4/idefics-80b": None,
    "HuggingFaceM4/idefics-80b-instruct": None,
}


class IDEFICSClient(CachingClient):
    """
    IDEFICS (Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS) is an
    open-access reproduction of Flamingo, a closed-source visual language model developed by Deepmind.
    Like GPT-4, the multimodal model accepts arbitrary sequences of image and text inputs and produces
    text outputs. IDEFICS is built solely on publicly available data and models.
    """

    END_OF_UTTERANCE_TOKEN: str = "<end_of_utterance>"
    BAD_WORD_TOKENS: List[str] = ["<image>", "<fake_token_around_image>"]

    def __init__(self, tokenizer: Tokenizer, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config, tokenizer=tokenizer)
        self._device: str = get_torch_device_name()

    def _get_model(self, checkpoint: str) -> LoadedIDEFICSModelProcessor:
        global _models_lock
        global _models

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[checkpoint]
            if loaded_model_processor is None:
                hlog(f"Loading model {checkpoint} and caching in memory...")
                model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(
                    self._device
                )
                processor = AutoProcessor.from_pretrained(checkpoint)
                _models[checkpoint] = LoadedIDEFICSModelProcessor(model, processor)
                loaded_model_processor = _models[checkpoint]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.model in _models, f"Not a valid model for this client: {request.model}"
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedIDEFICSModelProcessor = self._get_model(request.model)
        model = loaded_model_processor.model
        processor = loaded_model_processor.processor

        input_args: Dict[str, Union[str, bool]] = {"return_tensors": "pt"}
        generation_args = {
            "max_length": request.max_tokens,
            "bad_words_ids": processor.tokenizer(self.BAD_WORD_TOKENS, add_special_tokens=False).input_ids,
        }

        if self.END_OF_UTTERANCE_TOKEN in request.stop_sequences:
            # Following https://huggingface.co/HuggingFaceM4/idefics-80b-instruct,
            # specify <end_of_utterance> as an exit condition.
            input_args["add_end_of_utterance_token"] = False
            exit_condition = processor.tokenizer(self.END_OF_UTTERANCE_TOKEN, add_special_tokens=False).input_ids
            generation_args["eos_token_id"] = exit_condition

        multimodal_prompt: List[Union[str, Image.Image]] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                multimodal_prompt.append(open_image(media_object.location))
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                multimodal_prompt.append(media_object.text)
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
        prompt_text: str = request.multimodal_prompt.text.replace(self.END_OF_UTTERANCE_TOKEN, " ")

        try:

            def do_it():
                inputs = processor(multimodal_prompt, **input_args).to(self._device)
                generated_ids = model.generate(**inputs, **generation_args)
                generated_text: str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                assert generated_text.startswith(
                    prompt_text
                ), f"Generated text: {generated_text} does not start with prompt: {prompt_text}"

                # Remove the prompt from the generated text
                generated_text = generated_text[len(prompt_text) :].strip()
                return {"output": generated_text}

            # Include the prompt and model name in the cache key
            cache_key = CachingClient.make_cache_key(
                raw_request={
                    "model": request.model,
                    "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                    **generation_args,
                },
                request=request,
            )
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        # TODO: Support multiple completions and figure out how get the log probs
        # TODO: Does it make sense to support echo? Include these params in the cache key.
        # TODO: Together might support this model so use the TogetherClient
        tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
            TokenizationRequest(result["output"], tokenizer=request.model)
        )
        tokens: List[Token] = [
            Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
        ]
        completions: List[Sequence] = [Sequence(text=result["output"], logprob=0, tokens=tokens)]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
