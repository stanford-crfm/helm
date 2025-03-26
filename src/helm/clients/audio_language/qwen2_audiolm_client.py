from threading import Lock
import librosa
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedQwenModelProcessor:
    """Loaded model and processor for Qwen."""

    model: Qwen2AudioForConditionalGeneration
    tokenizer: AutoProcessor


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedQwenModelProcessor]] = {
    "Qwen/Qwen2-Audio-7B-Instruct": None,
}


class Qwen2AudioLMClient(CachingClient):
    """
    From https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct,
    Qwen2-Audio-Instruct (Qwen2 Large Vision Language Model) is the audito multimodal version of the large model series,
    Qwen2 (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen2-Audio-Instruct accepts audio, text as inputs,
    outputs text.
    Alibaba released Qwen-Audio and Qwen-Audio-Instruct, which is a instruction-following model based on Qwen-Audio.
    We for now integrated Qwen2-Audio-Instruct for instruction-following tasks.

    Paper: https://arxiv.org/abs/2407.10759
    """

    END_OF_TEXT_TOKEN: str = "<|im_end|>"

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model(self, helm_model_name: str) -> LoadedQwenModelProcessor:
        global _models_lock
        global _models

        model_name: str
        if helm_model_name == "qwen2-audio-7b-instruct":
            model_name = "Qwen/Qwen2-Audio-7B-Instruct"
        else:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[model_name]
            if loaded_model_processor is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=self._device,
                ).eval()
                tokenizer = AutoProcessor.from_pretrained(
                    model_name,
                )
                _models[model_name] = LoadedQwenModelProcessor(model, tokenizer)
                loaded_model_processor = _models[model_name]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedQwenModelProcessor = self._get_model(request.model_engine)
        model = loaded_model_processor.model
        tokenizer = loaded_model_processor.tokenizer

        input_query: List[Dict[str, Any]] = []
        query: List[Dict[str, str]] = []
        prompt_text: str = ""

        input_query.append({"role": "system", "content": "You are a helpful assistant."})
        prompt_text += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for media_num, media_object in enumerate(request.multimodal_prompt.media_objects):
            if media_object.is_type("audio") and media_object.location:
                assert media_object.is_local_file, "Only local audio files are supported"
                query.append({"type": "audio", "audio_url": media_object.location})

                prompt_text += f"<|im_start|>user\nAudio {media_num+1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                query.append({"type": "text", "text": media_object.text})
                prompt_text += media_object.text
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
        prompt_text += "<|im_end|>\n<|im_start|>assistant\n"

        input_query.append({"role": "user", "content": query})
        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {prompt_text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        inputs = tokenizer.apply_chat_template(input_query, add_generation_prompt=True, tokenize=False)
                        audios: List[Any] = []
                        # Refer to the official Qwen2-Audio documentation for the format of the input query
                        # https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
                        for message in input_query:
                            if isinstance(message["content"], list):
                                for element in message["content"]:
                                    if element["type"] == "audio":
                                        audios.append(
                                            librosa.load(
                                                element["audio_url"],
                                                sr=tokenizer.feature_extractor.sampling_rate,
                                            )[0]
                                        )
                        inputs = tokenizer(
                            text=inputs,
                            audios=audios,
                            sampling_rate=tokenizer.feature_extractor.sampling_rate,
                            return_tensors="pt",
                            padding=True,
                        )
                        input_length = inputs.input_ids.size(1)
                        # Qwen2-Audio-Instruct counts input into the max_length,
                        # so we need to add the length of the prompt
                        inputs = inputs.to(self._device)
                        pred = model.generate(**inputs, max_length=request.max_tokens + input_length)[:, input_length:]

                        completion = tokenizer.decode(
                            pred.cpu()[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # The processor of Qwen2-Audio-Instruct consists an AutoTokenizer and a WhisperFeatureExtractor
                        tokens: List[str] = tokenizer.tokenizer.tokenize(completion)
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
