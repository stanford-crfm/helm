from threading import Lock
from typing import List

import torch
from huggingface_hub import hf_hub_download
from helm.proxy.clients.vision_language.open_flamingo import create_model_and_transforms

from helm.common.cache import CacheConfig
from helm.common.images_utils import open_image
from helm.common.gpu_utils import get_torch_device_name
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.request import wrap_request_time
from helm.proxy.clients.client import CachingClient, generate_uid_for_multimodal_prompt

try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])


class OpenFlamingoClient(CachingClient):
    """
    OpenFlamingo is an open source implementation of DeepMind's Flamingo models.
    https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b
    """

    END_OF_CHUNK_TOKEN: str = "<|endofchunk|>"
    IMAGE_TOKEN: str = "<image>"

    _model_lock: Lock = Lock()

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config)
        self._device: str = get_torch_device_name()
        self._get_model()

    def _get_model(self):
        with self._model_lock:
            self._model, self.image_processor, self.tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-7b",
                tokenizer_path="anas-awadalla/mpt-7b",
                cross_attn_every_n_layers=4,
            )
            self.tokenizer.padding_side = "left"
            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
            self._model.load_state_dict(torch.load(checkpoint_path), strict=False)
            self._model = self._model.to(self._device)

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        # Build the prompt
        prompt_text: str = ""
        images: List[Image.Image] = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                images.append(open_image(media_object.location))
                prompt_text += self.IMAGE_TOKEN
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                prompt_text += media_object.text + self.END_OF_CHUNK_TOKEN
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        # Preprocess
        vision_x: torch.Tensor = torch.cat([self.image_processor(image).unsqueeze(0) for image in images], dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        lang_x = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
        )

        # Generate
        try:
            generation_args = {
                "max_new_tokens": request.max_tokens,
                "num_beams": 1,
            }

            def do_it():
                generated_text: str = self._model.generate(
                    vision_x=vision_x.to(self._device),
                    lang_x=lang_x["input_ids"].to(self._device),
                    attention_mask=lang_x["attention_mask"].to(self._device),
                    max_new_tokens=generation_args["max_new_tokens"],
                    num_beams=generation_args["num_beams"],
                )
                generated_text = self.tokenizer.decode(generated_text[0])
                assert generated_text.startswith(
                    prompt_text
                ), f"Generated text: {generated_text} does not start with prompt: {prompt_text}"

                # Remove the prompt from the generated text
                generated_text = generated_text[len(prompt_text) :].strip()
                return {"output": generated_text}

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

        tokens: List[Token] = [
            Token(text=str(self.tokenizer.decode(id)), logprob=0, top_logprobs={}) for id in lang_x["input_ids"][0]
        ]
        completions: List[Sequence] = [Sequence(text=result["generated_text"], logprob=0, tokens=tokens)]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=result["request_time"],
            completions=completions,
            embedding=[],
        )
