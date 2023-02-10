from typing import List, Dict, Optional
import base64
import requests
from helm.common.hierarchical_logger import hlog

from transformers import AutoModelForCausalLM, AutoTokenizer

from helm.common.cache import CacheConfig
from helm.common.file_caches.file_cache import FileCache
from helm.common.request import Request, RequestResult, Sequence, TextToImageRequest
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)

from .client import Client, wrap_request_time
from .together_client import TogetherClient


class TogetherVisionClient(TogetherClient):
    """
    Client for image generation via the Together API.
    """

    DEFAULT_IMAGE_HEIGHT: int = 512
    DEFAULT_IMAGE_WIDTH: int = 512

    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_STEPS: int = 50

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache, api_key: Optional[str] = None):
        super().__init__(cache_config, api_key)
        self.file_cache: FileCache = file_cache

        self._promptist_model = None
        self._promptist_tokenizer = None

    def make_request(self, request: Request) -> RequestResult:
        assert isinstance(request, TextToImageRequest)
        # Following https://docs.together.xyz/en/api
        # TODO: support the four different SafeStableDiffusion configurations
        raw_request = {
            "request_type": "image-model-inference",
            "model": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
            "guidance_scale": request.guidance_scale
            if request.guidance_scale is not None
            else self.DEFAULT_GUIDANCE_SCALE,
            "steps": request.steps if request.steps is not None else self.DEFAULT_STEPS,
        }

        if request.width is None or request.height is None:
            raw_request["width"] = self.DEFAULT_IMAGE_WIDTH
            raw_request["height"] = self.DEFAULT_IMAGE_HEIGHT
        else:
            raw_request["width"] = request.width
            raw_request["height"] = request.height

        cache_key: Dict = Client.make_cache_key(raw_request, request)

        try:

            def do_it():
                if request.model_engine == "PromptistStableDiffusion":
                    promptist_request = dict(raw_request)
                    promptist_request["model"] = "StableDiffusion"
                    promptist_prompt: str = self._generate_promptist_version(promptist_request["prompt"])
                    hlog(f"{promptist_request['prompt']} -> {promptist_prompt}")
                    promptist_request["prompt"] = promptist_prompt
                    result = requests.post(self.INFERENCE_ENDPOINT, json=promptist_request).json()
                else:
                    result = requests.post(self.INFERENCE_ENDPOINT, json=raw_request).json()
                assert "output" in result, f"Invalid response: {result} from prompt: {request.prompt}"

                for choice in result["output"]["choices"]:
                    # Write out the image to a file and save the path
                    choice["file_path"] = self.file_cache.store(lambda: base64.b64decode(choice["image_base64"]))
                    choice.pop("image_base64", None)
                return result["output"]

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"TogetherVisionClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(text="", logprob=0, tokens=[], file_location=choice["file_path"]) for choice in response["choices"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            completions=completions,
            embedding=[],
        )

    def _generate_promptist_version(self, prompt: str) -> str:
        """
        Generate a better version of the prompt with Promptist.
        Adapted from https://huggingface.co/spaces/microsoft/Promptist/blob/main/app.py.
        """

        def load_promptist():
            prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            return prompter_model, tokenizer

        def generate(plain_text: str) -> str:
            if self._promptist_model is None or self._promptist_tokenizer is None:
                self._promptist_model, self._promptist_tokenizer = load_promptist()
            assert self._promptist_model is not None
            assert self._promptist_tokenizer is not None

            input_ids = self._promptist_tokenizer(f"{plain_text.strip()} Rephrase:", return_tensors="pt").input_ids
            eos_id = self._promptist_tokenizer.eos_token_id
            # Used the same hyperparameters from the example
            outputs = self._promptist_model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=75,
                num_beams=8,
                num_return_sequences=8,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                length_penalty=-1.0,
            )
            output_texts: List[str] = self._promptist_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for output_text in output_texts:
                res: str = output_text.replace(f"{plain_text} Rephrase:", "").strip()
                # The Promptist model sometimes generates empty string results.
                # Return the first non-empty string result.
                if len(res) > 0:
                    return res

            # If all fails, just return the original text.
            return plain_text

        return generate(prompt)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
