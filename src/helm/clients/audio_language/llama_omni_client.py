from threading import Lock
import torch
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from transformers import AutoTokenizer
import whisper
from helm.clients.audio_language.llama_omni.model.builder import load_pretrained_model as load_llama_omni
from helm.clients.audio_language.llama_omni.model.language_model.omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM
from helm.clients.audio_language.llama_omni.conversation import conv_templates, Conversation
from helm.clients.audio_language.llama_omni.preprocess import tokenizer_speech_token

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedLlamaOmniModelProcessor:
    """Loaded model and processor for Qwen."""

    model: OmniSpeech2SLlamaForCausalLM
    tokenizer: AutoTokenizer


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedLlamaOmniModelProcessor]] = {
    "ICTNLP/Llama-3.1-8B-Omni": None,
}


class LlamaOmniAudioLMClient(CachingClient):
    """
    From https://github.com/ictnlp/LLaMA-Omni,
    LLaMA-Omni is the audio multimodal version based on the LLaMA-3.1-8B large language model,
    developed by ICTNLP group. LLaMA-Omni accepts audio, text as inputs, and outputs text.

    Paper: https://arxiv.org/abs/2409.06666
    """

    END_OF_TEXT_TOKEN: str = "<|im_end|>"
    CONV_MODE: str = "llama_3"
    PAD_ID: int = 128004
    MEL_NUM: int = 128

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model(self, helm_model_name: str) -> LoadedLlamaOmniModelProcessor:
        global _models_lock
        global _models

        model_name: str
        if helm_model_name == "llama-3.1-8b-omni":
            model_name = "ICTNLP/Llama-3.1-8B-Omni"
        else:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[model_name]
            if loaded_model_processor is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                # Follow the official LLaMA-Omni model loading pattern:
                # https://github.com/ictnlp/LLaMA-Omni/blob/main/omni_speech/infer/run.sh
                tokenizer, model, _ = load_llama_omni(model_name, None, s2s=True)
                _models[model_name] = LoadedLlamaOmniModelProcessor(model, tokenizer)
                loaded_model_processor = _models[model_name]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def _load_local_audio(self, media_object) -> torch.Tensor:
        assert media_object.is_local_file, "LLaMA-Omni only supports local audio file input"
        audio_media = whisper.load_audio(media_object.location)
        audio_media = whisper.pad_or_trim(audio_media)
        audio_media = whisper.log_mel_spectrogram(audio_media, n_mels=self.MEL_NUM).permute(1, 0)
        return audio_media

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedLlamaOmniModelProcessor = self._get_model(request.model_engine)
        model = loaded_model_processor.model
        tokenizer = loaded_model_processor.tokenizer

        # The generation configs are taken from the official LLaMA-Omni repository
        # https://github.com/ictnlp/LLaMA-Omni/blob/main/omni_speech/infer/infer.py#L116
        generation_args = {
            "max_new_tokens": 25,
            "do_sample": False,
            "use_cache": False,
            "pad_token_id": self.PAD_ID,
            "streaming_unit_gen": False,
            "top_p": None,
        }

        input_text_query: Dict[str, str]
        input_audio_query: Dict[str, Any]
        prompt_text: str = ""

        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("audio") and media_object.location:
                input_audio_query = {"audio": self._load_local_audio(media_object)}
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                input_text_query = {"text": "<speech>\n" + media_object.text}
                prompt_text += media_object.text
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {prompt_text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        conv: Conversation = conv_templates[self.CONV_MODE].copy()
                        conv.append_message(conv.roles[0], input_text_query["text"])
                        conv.append_message(conv.roles[1], None)
                        query: str = conv.get_prompt()
                        # LLama-Omni requires a batch input
                        text_inputs = (
                            tokenizer_speech_token(query, tokenizer, return_tensors="pt").unsqueeze(0).to(self._device)
                        )
                        audio_inputs = (
                            input_audio_query["audio"].to(dtype=torch.float16, device=self._device).unsqueeze(0)
                        )
                        speech_length = torch.LongTensor([audio_inputs.shape[1]])
                        pred, _ = model.generate(
                            text_inputs,
                            audio_inputs,
                            speech_length,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            False,
                            None,
                            None,
                            **generation_args,
                        )
                        completion = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
                        tokens: List[str] = tokenizer.tokenize(completion)
                        return {"output": (completion, tokens)}

                    # Include the prompt and model name in the cache key
                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "completion_index": completion_index,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                            **generation_args,
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                except RuntimeError as model_error:
                    return RequestResult(
                        success=False, cached=False, error=str(model_error), completions=[], embedding=[]
                    )

                text, tokens = result["output"]

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
