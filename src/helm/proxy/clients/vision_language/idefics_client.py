from typing import Dict, List, Optional, Tuple, Union

import torch
from dataclasses import asdict
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, IdeficsProcessor

from helm.common.cache import CacheConfig, Cache
from helm.common.file import File
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request, RequestResult, Sequence
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.proxy.clients.client import Client, wrap_request_time, cleanup_tokens


class IDEFICSClient(Client):
    """
    IDEFICS (Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS) is an
    open-access reproduction of Flamingo, a closed-source visual language model developed by Deepmind.
    Like GPT-4, the multimodal model accepts arbitrary sequences of image and text inputs and produces
    text outputs. IDEFICS is built solely on publicly available data and models.
    """

    END_OF_UTTERANCE_TOKEN: str = "<end_of_utterance>"
    BAD_WORD_TOKENS: List[str] = ["<image>", "<fake_token_around_image>"]

    def __init__(self, cache_config: CacheConfig):
        self._cache = Cache(cache_config)
        self._device: str = get_torch_device_name()
        self._models: Dict[str, Tuple[Optional[IdeficsForVisionText2Text], Optional[AutoProcessor]]] = {
            "HuggingFaceM4/idefics-9b": (None, None),
            "HuggingFaceM4/idefics-9b-instruct": (None, None),
            "HuggingFaceM4/idefics-80b": (None, None),
            "HuggingFaceM4/idefics-80b-instruct": (None, None),
        }

    def _get_model(self, checkpoint: str) -> Tuple[IdeficsForVisionText2Text, IdeficsProcessor]:
        model, processor = self._models[checkpoint]
        if model is None or processor is None:
            hlog(f"Loading model {checkpoint} and caching...")
            model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(self._device)
            processor = AutoProcessor.from_pretrained(checkpoint)
            self._models[checkpoint] = (model, processor)
        return model, processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.model in self._models, f"Not a valid model for this client: {request.model}"
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"
        model, processor = self._get_model(request.model)

        input_args: Dict[str, Union[str, bool]] = {"return_tensors": "pt"}
        generation_args = {
            "max_length": request.max_tokens,
            "bad_words_ids": processor.tokenizer(self.BAD_WORD_TOKENS, add_special_tokens=False).input_ids,
        }
        if self.END_OF_UTTERANCE_TOKEN in request.stop_sequences:
            input_args["add_end_of_utterance_token"] = False
            exit_condition = processor.tokenizer(self.END_OF_UTTERANCE_TOKEN, add_special_tokens=False).input_ids
            generation_args["eos_token_id"] = exit_condition

        multimodal_prompt: List[Union[str, Image]] = [
            open_image(content.location) if isinstance(content, File) else content
            for content in request.multimodal_prompt.content
        ]
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
            cache_key = Client.make_cache_key(
                {"model": request.model, "prompt": str(request.multimodal_prompt), **generation_args}, request
            )
            result, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        completions: List[Sequence] = [Sequence(text=result["output"], logprob=0, tokens=[])]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        _, processor = self._get_model(request.tokenizer)
        tokenizer = processor.tokenizer
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    tokens = tokenizer.tokenize(request.text)
                    tokens = cleanup_tokens(tokens, request.tokenizer)
                return {"tokens": tokens}

            result, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace tokenize error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        _, processor = self._get_model(request.tokenizer)
        tokenizer = processor.tokenizer
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace decode error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
