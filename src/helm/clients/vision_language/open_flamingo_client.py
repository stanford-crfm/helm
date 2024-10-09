from threading import Lock
from typing import List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device_name
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.vision_language.open_flamingo import create_model_and_transforms
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])


class OpenFlamingoClient(CachingClient):
    """
    OpenFlamingo is an open source implementation of DeepMind's Flamingo models.
    Implementation following:
        https://github.com/mlfoundations/open_flamingo
        https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b
    """

    END_OF_CHUNK_TOKEN: str = "<|endofchunk|>"
    IMAGE_TOKEN: str = "<image>"

    _model_lock: Lock = Lock()

    def __init__(
        self,
        cache_config: CacheConfig,
        checkpoint_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        cross_attn_every_n_layers: int = 4,
    ):
        super().__init__(cache_config)
        self._device: str = get_torch_device_name()
        self._checkpoint_path: Optional[str] = checkpoint_path
        self._tokenizer_name: Optional[str] = tokenizer_name
        self._cross_attn_every_n_layers: int = cross_attn_every_n_layers

        # Model
        # The model is only initialized when the first request is made
        # This is to avoid loading the model if it is not used
        self._model: Optional[torch.nn.Module] = None

    def _get_model(self):
        if not self._checkpoint_path:
            raise ValueError("OpenFlamingoClient requires a checkpoint path")
        if not self._tokenizer_name:
            raise ValueError("OpenFlamingoClient requires a tokenizer name")
        with htrack_block("Initializing OpenFlamingo model"):
            with self._model_lock:
                self._model, self.image_processor, self.tokenizer = create_model_and_transforms(
                    clip_vision_encoder_path="ViT-L-14",
                    clip_vision_encoder_pretrained="openai",
                    lang_encoder_path=self._tokenizer_name,
                    tokenizer_path=self._tokenizer_name,
                    cross_attn_every_n_layers=self._cross_attn_every_n_layers,
                )
                self.tokenizer.padding_side = "left"
                checkpoint_path = hf_hub_download(self._checkpoint_path, "checkpoint.pt")
                self._model.load_state_dict(torch.load(checkpoint_path), strict=False)
                self._model = self._model.to(self._device)
                hlog(f"Loaded model to {self._device}.")

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        # Load model if needed
        if self._model is None:
            self._get_model()

        # Build the prompt
        prompt_text: str = ""
        images: List[Image.Image] = []
        request.validate()
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                images.append(open_image(media_object.location))
                prompt_text += self.IMAGE_TOKEN
            elif media_object.is_type(TEXT_TYPE):
                prompt_text += media_object.text
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        # Preprocess
        vision_x: torch.Tensor = torch.cat([self.image_processor(image).unsqueeze(0) for image in images], dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = self.tokenizer([prompt_text], return_tensors="pt")

        # Generate
        try:
            generation_args = {
                "max_new_tokens": request.max_tokens,
                "n": request.num_completions,
            }

            def do_it():
                tensors = self._model.generate(
                    vision_x=vision_x.to(self._device),
                    lang_x=lang_x["input_ids"].to(self._device),
                    attention_mask=lang_x["attention_mask"].to(self._device),
                    max_new_tokens=generation_args["max_new_tokens"],
                    num_beams=generation_args["n"],
                    num_return_sequences=generation_args["n"],
                )
                generated_completions: List[Tuple[str, List[str]]] = []
                for tensor in tensors:
                    generated_text: str = self.tokenizer.decode(tensor)
                    raw_tokens: List[str] = self.tokenizer.tokenize(generated_text)
                    generated_completions.append((generated_text, raw_tokens))

                return {"output": generated_completions}

            cache_key = CachingClient.make_cache_key(
                raw_request={
                    "model": request.model,
                    "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                    **generation_args,
                },
                request=request,
            )
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as ex:
            return RequestResult(success=False, cached=False, error=str(ex), completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for text, tokens in result["output"]:
            # Remove the prompt from the generated text
            text = (
                text[len(prompt_text) :].replace(self.END_OF_CHUNK_TOKEN, "").strip()
                if len(text) >= len(prompt_text)
                else text[-1]
            )
            completions.append(
                GeneratedOutput(text=text, logprob=0, tokens=[Token(text=token, logprob=0) for token in tokens])
            )

        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
