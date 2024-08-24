from threading import Lock
from typing import Dict, Optional, List

from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForCausalLM

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog
from helm.common.images_utils import open_image
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.clients.client import CachingClient


try:
    from PIL import Image
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["images"])


@dataclass(frozen=True)
class PhiVisionModelProcessor:
    """Loaded model and processor for Phi Vision model."""

    model: AutoModelForCausalLM
    processor: AutoProcessor


_phi_models_lock: Lock = Lock()
_phi_models: Dict[str, Optional[PhiVisionModelProcessor]] = {
    "microsoft/Phi-3.5-vision-instruct": None,
}


class HuggingFacePhiVisionClient(CachingClient):
    """
    Client for Phi Vision models from HuggingFace.
    """

    def __init__(self, tokenizer_name: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.tokenizer_name = tokenizer_name
        self._device: str = get_torch_device_name()

    def _get_model(self, checkpoint: str) -> PhiVisionModelProcessor:
        global _phi_models_lock
        global _phi_models

        with _phi_models_lock:
            loaded_model_processor = _phi_models[checkpoint]
            if loaded_model_processor is None:
                # Following https://huggingface.co/microsoft/Phi-3.5-vision-instruct
                hlog(f"Loading model {checkpoint} and caching in memory...")
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    device_map=self._device,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    _attn_implementation="eager",
                )
                processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, num_crops=4)

                _phi_models[checkpoint] = PhiVisionModelProcessor(model, processor)
                loaded_model_processor = _phi_models[checkpoint]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.model_deployment in _phi_models, f"Not a valid model for this client: {request.model_deployment}"
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: PhiVisionModelProcessor = self._get_model(request.model_deployment)
        model = loaded_model_processor.model
        processor = loaded_model_processor.processor

        images: List[Image.Image] = []
        placeholder: str = ""

        # Assuming `media_objects` contains the URLs or file paths to the images
        # All images should be placed before the text in the prompt
        for i, media_object in enumerate(request.multimodal_prompt.media_objects, start=1):
            if media_object.is_type("image") and media_object.location:
                image_url = media_object.location
                images.append(open_image(image_url))
                placeholder += f"<|image_{i}|>\n"
        placeholder += request.multimodal_prompt.text

        messages = [
            {"role": "user", "content": placeholder},
        ]
        hlog(placeholder)

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(prompt, images, return_tensors="pt").to(self._device)

        generation_args = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "do_sample": False,
        }

        try:
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
            generated_texts = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        except RuntimeError as model_error:
            return RequestResult(success=False, cached=False, error=str(model_error), completions=[], embedding=[])

        completions = [GeneratedOutput(text=text, logprob=0, tokens=[]) for text in generated_texts]

        return RequestResult(
            success=True,
            cached=False,
            request_time=None,
            completions=completions,
            embedding=[],
        )
