from threading import Lock
import torch
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from helm.clients.audio_language.qwen_omni.modeling_qwen2_5_omni import Qwen2_5OmniModel
from helm.clients.audio_language.qwen_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from helm.clients.audio_language.qwen_omni.qwen2_5_omni_utils.v2_5 import process_mm_info

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedQwen2_5OmniModelProcessor:
    """Loaded model and processor for Qwen."""

    model: Qwen2_5OmniModel
    tokenizer: Qwen2_5OmniProcessor


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedQwen2_5OmniModelProcessor]] = {
    "Qwen/Qwen2.5-Omni-7B": None,
}


class Qwen2_5OmniAudioLMClient(CachingClient):
    """
    From https://huggingface.co/Qwen/Qwen2.5-Omni-7B,
    Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text,
    images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner.

    Paper: https://arxiv.org/abs/2503.20215
    """

    END_OF_TEXT_TOKEN: str = "<|endoftext|>>"

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model(self, helm_model_name: str) -> LoadedQwen2_5OmniModelProcessor:
        global _models_lock
        global _models

        model_name: str
        if helm_model_name == "qwen2.5-omni-7b":
            model_name = "Qwen/Qwen2.5-Omni-7B"
        else:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[model_name]
            if loaded_model_processor is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                model = Qwen2_5OmniModel.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map=self._device,
                ).eval()
                tokenizer = Qwen2_5OmniProcessor.from_pretrained(
                    model_name,
                )
                _models[model_name] = LoadedQwen2_5OmniModelProcessor(model, tokenizer)
                loaded_model_processor = _models[model_name]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedQwen2_5OmniModelProcessor = self._get_model(request.model_engine)
        model = loaded_model_processor.model
        tokenizer = loaded_model_processor.tokenizer

        input_query: List[Dict[str, Any]] = []
        query: List[Dict[str, str]] = []
        prompt_text: str = ""

        input_query.append(
            {
                "role": "system",
                "content": (
                    "You are Qwen, a virtual human developed by the Qwen Team,"
                    " Alibaba Group, capable of perceiving auditory and visual inputs,"
                    " as well as generating text and speech."
                ),
            }
        )
        # prompt_text += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for media_num, media_object in enumerate(request.multimodal_prompt.media_objects):
            if media_object.is_type("audio") and media_object.location:
                assert media_object.is_local_file, "Only local audio files are supported"
                query.append({"type": "audio", "audio": media_object.location})

                # prompt_text += f"<|im_start|>user\nAudio {media_num+1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                query.append({"type": "text", "text": media_object.text})
                # prompt_text += media_object.text
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
        # prompt_text += "<|im_end|>\n<|im_start|>assistant\n"

        input_query.append({"role": "user", "content": query})

        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {prompt_text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        # Refer to the official Qwen2.5-Omni documentation for the format of the input query
                        # https://huggingface.co/Qwen/Qwen2.5-Omni-7B
                        USE_AUDIO_IN_VIDEO = True
                        text = tokenizer.apply_chat_template(input_query, add_generation_prompt=True, tokenize=False)
                        audios, images, videos = process_mm_info(input_query, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                        inputs = tokenizer(
                            text=text,
                            audios=audios,
                            images=images,
                            videos=videos,
                            return_tensors="pt",
                            padding=True,
                            use_audio_in_video=USE_AUDIO_IN_VIDEO,
                        )
                        inputs = inputs.to(self._device, torch.bfloat16)
                        input_seq_length = len(inputs.input_ids[0])
                        # The model runs into errors when setting thinker_max_new_tokens to 1
                        if request.max_tokens != 1:
                            pred, _ = model.generate(**inputs, thinker_max_new_tokens=request.max_tokens)
                            pred_decode = pred.cpu()[0][input_seq_length:]
                        else:
                            pred, _ = model.generate(**inputs)
                            pred_decode = pred.cpu()[0][input_seq_length : input_seq_length + 1]
                        completion = tokenizer.decode(
                            pred_decode,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        # The processor of Qwen2-Audio-Instruct consists an AutoTokenizer and a WhisperFeatureExtractor
                        tokens: List[str] = tokenizer.tokenizer.tokenize(completion)  # type: ignore
                        return {"output": (completion, tokens)}

                    # Include the prompt and model name in the cache key
                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "completion_index": completion_index,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                            "max_tokens": request.max_tokens,
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                except RuntimeError as model_error:
                    return RequestResult(
                        success=False, cached=False, error=str(model_error), completions=[], embedding=[]
                    )

                text, tokens = result["output"]
                hlog(f"Generated: {text}")

                # Tokenize truncated text to get the list of tokens
                completions.append(
                    GeneratedOutput(
                        text=text, logprob=0, tokens=[Token(text=str(token), logprob=0) for token in tokens]
                    )
                )

                request_time += result["request_time"]
                # Use the datetime from the first completion because that's when the request was fired
                request_datetime = request_datetime or result.get("request_datetime")
                all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
