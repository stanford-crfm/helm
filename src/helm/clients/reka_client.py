# mypy: check_untyped_defs = False
import requests
from typing import Any, Dict, List, Optional, TypedDict

from helm.proxy.retry import NonRetriableException
from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.tokenizers.tokenizer import Tokenizer
from helm.clients.client import CachingClient, truncate_and_tokenize_response_text

try:
    import reka
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["reka-api"])


class RekaAIRequest(TypedDict):
    """Data passed between make_request and _send_request. Used as the cache key."""

    model_name: str
    conversation_history: List[Dict[str, str]]
    request_output_len: int
    temperature: float
    runtime_top_p: float
    random_seed: Optional[int]
    stop_words: Optional[List[str]]
    presence_penalty: float
    frequency_penalty: float


class RekaClient(CachingClient):
    REKA_CHAT_ROLE_MAPPING: Dict[str, str] = {
        "user": "human",
        "assistant": "model",
    }

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.client = reka
        self.client.API_KEY = api_key

    def _is_reka_model_engine(self, model_engine: str) -> bool:
        if (
            model_engine.startswith("reka-edge")
            or model_engine.startswith("reka-flash")
            or model_engine.startswith("reka-core")
        ):
            return True
        else:
            return False

    def _get_model_for_request(self, request: Request) -> str:
        return request.model_engine

    def _get_random_seed(self, request: Request, completion_index: int) -> Optional[int]:
        if request.random is None and completion_index == 0:
            return None

        # Treat the user's request.random as an integer for the random seed.
        try:
            request_random_seed = int(request.random) if request.random is not None else 0
        except ValueError:
            raise NonRetriableException("RekaAIClient only supports integer values for request.random")

        # A large prime is used so that the resulting values are unlikely to collide
        # with request.random values chosen by the user.
        fixed_large_prime = 1911011
        completion_index_random_seed = completion_index * fixed_large_prime

        return request_random_seed + completion_index_random_seed

    def _convert_messages_to_reka_chat_history(self, messages: List[Dict[str, Any]]):
        chat_history = []
        num_images: int = 0
        for chat_turn, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            current_chat_history: Dict[str, Any] = {
                "type": self.REKA_CHAT_ROLE_MAPPING[role],
                "text": "",  # text placeholder
                "media_url": None,
            }
            for item in content:
                if item["type"] == "image_url":
                    if chat_turn == 0 and num_images == 0:
                        current_chat_history["media_url"] = item["image_url"]["url"]
                        num_images += 1
                    else:
                        raise ValueError(
                            f"Only the first message can contain one image. Found image input "
                            f"in message {chat_turn + 1}"
                        )
                elif item["type"] == "text":
                    current_chat_history["text"] = item["text"]
                else:
                    raise ValueError(f"Unrecognized message type {item['type']}")
            chat_history.append(current_chat_history)
        return chat_history

    def make_request(self, request: Request) -> RequestResult:
        completions: List[GeneratedOutput] = []
        messages: Optional[List[Dict[str, Any]]] = request.messages
        reka_chat_history: List[Dict[str, Any]]
        if messages is not None:
            # Checks that all messages have a role and some content
            for message in messages:
                if not message.get("role") or not message.get("content"):
                    raise ValueError("All messages must have a role and content")
            # Checks that the last role is "user"
            if messages[-1]["role"] != "user":
                raise ValueError("Last message must have role 'user'")
            if request.prompt != "":
                hlog("WARNING: Since message is set, prompt will be ignored")
            reka_chat_history = self._convert_messages_to_reka_chat_history(messages)
        else:
            current_chat_history: Dict[str, Any] = {
                "type": "human",
                "text": "",
                "media_url": None,
            }
            if request.multimodal_prompt is not None:
                for media_object in request.multimodal_prompt.media_objects:
                    if media_object.is_type("image") and media_object.location:
                        from helm.common.images_utils import encode_base64

                        base64_image: str = encode_base64(media_object.location)
                        current_chat_history["media_url"] = f"data:image/jpeg;base64,{base64_image}"
                    elif media_object.is_type(TEXT_TYPE):
                        if media_object.text is None:
                            raise ValueError("MediaObject of text type has missing text field value")
                        current_chat_history["text"] = media_object.text
                    else:
                        raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

            else:
                current_chat_history["text"] = request.prompt
            reka_chat_history = [current_chat_history]

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:
                raw_request: RekaAIRequest = {
                    "model_name": self._get_model_for_request(request),
                    "conversation_history": reka_chat_history,  # we only use chat_history as the input
                    "request_output_len": request.max_tokens,
                    "temperature": request.temperature,
                    "random_seed": self._get_random_seed(request, completion_index),
                    "stop_words": request.stop_sequences or None,  # API doesn't like empty list
                    "runtime_top_p": request.top_p,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                }

                def do_it() -> Dict[str, Any]:
                    return self.client.chat(**raw_request)

                response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"RekaClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            response_message: Dict[str, Any] = response
            assert response_message["type"] == "model"
            response_text: str = response_message["text"]

            # The Reka API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            completion = truncate_and_tokenize_response_text(text, request, self.tokenizer, self.tokenizer_name)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )
