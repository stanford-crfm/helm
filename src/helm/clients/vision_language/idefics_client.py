from threading import Lock
from typing import Any, Dict, List, Optional, Union

import torch
from dataclasses import dataclass
from transformers import IdeficsForVisionText2Text, AutoProcessor, IdeficsProcessor

from helm.common.cache import CacheConfig
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.tokenization_request import TokenizationRequest
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt
from helm.tokenizers.tokenizer import Tokenizer

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

    ASSISTANT_PREFIX: str = "Assistant: "

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self._device: str = get_torch_device_name()

    def _get_model(self, checkpoint: str) -> LoadedIDEFICSModelProcessor:
        global _models_lock
        global _models

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[checkpoint]
            if loaded_model_processor is None:
                hlog(f"Loading model {checkpoint} and caching in memory...")
                model = IdeficsForVisionText2Text.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(checkpoint)
                _models[checkpoint] = LoadedIDEFICSModelProcessor(model, processor)
                loaded_model_processor = _models[checkpoint]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.model_deployment in _models, f"Not a valid model for this client: {request.model_deployment}"
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedIDEFICSModelProcessor = self._get_model(request.model_deployment)
        model = loaded_model_processor.model
        processor = loaded_model_processor.processor

        input_args: Dict[str, Union[str, bool]] = {"return_tensors": "pt"}
        generation_args = {
            "max_new_tokens": request.max_tokens,
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

        completions: List[GeneratedOutput] = []
        with htrack_block(f"Generating for prompt: {prompt_text}"):
            try:

                def do_it() -> Dict[str, Any]:
                    inputs = processor([multimodal_prompt] * request.num_completions, **input_args).to(self._device)
                    generated_ids = model.generate(**inputs, **generation_args)
                    generated_text: List[str] = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    return {"output": generated_text}

                # Include the prompt and model name in the cache key
                cache_key = CachingClient.make_cache_key(
                    raw_request={
                        "n": request.num_completions,
                        "model": request.model,
                        "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                        **generation_args,
                    },
                    request=request,
                )
                result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except RuntimeError as model_error:
                return RequestResult(success=False, cached=False, error=str(model_error), completions=[], embedding=[])

            for text in result["output"]:
                hlog(f"Generated text: {text}")

                # Truncate the output text as IDEFICS outputs the entire sequence including the prompt
                if "instruct" in request.model:
                    assert self.ASSISTANT_PREFIX in text, f"Expected {self.ASSISTANT_PREFIX} in the output: {text}"
                    text = text.rpartition(self.ASSISTANT_PREFIX)[-1]
                else:
                    # Best we can do is to remove the text portion of the prompt from the output
                    text = text[len(prompt_text) :]

                # Tokenize truncated text to get the list of tokens
                hlog(f"Truncated: {text}")
                tokenization_result = self.tokenizer.tokenize(
                    TokenizationRequest(text, tokenizer=self.tokenizer_name, encode=False)
                )
                tokens: List[Token] = [Token(text=str(text), logprob=0) for text in tokenization_result.raw_tokens]
                completions.append(GeneratedOutput(text=text, logprob=0, tokens=tokens))

        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
