import requests
from typing import List, Optional, TypedDict

from helm.proxy.retry import NonRetriableException
from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, MultimediaObject
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.media_object import MediaObject
from helm.clients.client import CachingClient, truncate_and_tokenize_response_text
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import reka
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["reka"])


class RekaAIRequest(TypedDict):
    """Data passed between make_request and _send_request. Used as the cache key."""

    model_name: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k_per_token: int
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: List[str]
    multimodal_prompt: Optional[MultimediaObject]
    random: Optional[int]


class RekaAIClient(CachingClient):
    """
    Client for the Google models. There isn't an API for their language models.
    We receive and process completions offline.
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str, cache_config: CacheConfig, api_key: str):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        reka.API_KEY = api_key

    def _get_random_seed(self, request: Request, completion_index: int) -> Optional[int]:
        if request.random is None and completion_index == 0:
            return None

        # Treat the user's request.random as an integer for the random seed.
        try:
            request_random_seed = int(request.random) if request.random is not None else 0
        except ValueError:
            raise NonRetriableException("MistralAIClient only supports integer values for request.random")

        # A large prime is used so that the resulting values are unlikely to collide
        # with request.random values chosen by the user.
        fixed_large_prime = 1911011
        completion_index_random_seed = completion_index * fixed_large_prime

        return request_random_seed + completion_index_random_seed

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        completions: List[GeneratedOutput] = []

        # `num_completions` is not supported, so instead make `num_completions` separate requests.
        for completion_index in range(request.num_completions):
            try:
                random: Optional[int] = self._get_random_seed(request, completion_index)
                cache_key: RekaAIRequest = {
                    "model_name": request.model_engine,
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k_per_token": request.top_k_per_token,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty,
                    "stop_sequences": request.stop_sequences,
                    "multimodal_prompt": request.multimodal_prompt,
                    "random": random,
                }

                def do_it():
                    media_filename: Optional[str] = None
                    media_url: Optional[str] = None

                    if request.multimodal_prompt:
                        media_objects: List[MediaObject] = request.multimodal_prompt.media_objects
                        if len(media_objects) > 1:
                            raise RuntimeError("Reka only supports one media object in multimodal prompts")
                        media_object: MediaObject = media_objects[0]

                        if media_object.is_local_file:
                            media_filename = media_object.location
                        else:
                            media_url = media_object.location

                    response = reka.chat(
                        human=request.prompt,
                        model_name=request.model_engine,
                        request_output_len=request.max_tokens,
                        temperature=request.temperature,
                        random_seed=random,
                        runtime_top_p=request.top_p,
                        runtime_top_k=request.top_k_per_token,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        stop_words=request.stop_sequences,
                        # Multimodal prompt
                        media_filename=media_filename,
                        media_url=media_url,
                    )
                    assert "type" in response, "Response does not contain 'type' field"
                    assert response["type"] == "model", "Response type is not 'model'"
                    assert "text" in response, "Response does not contain 'text' field"
                    return response

                # If results are not cached for a given query, fail fast
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            except (requests.exceptions.RequestException, AssertionError) as e:
                error: str = f"RekaClient error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            # Expect the result to be structured the same way as a response from OpenAI API.
            response_text: str = response["text"]

            # The Reka API doesn't support echo. If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + response_text if request.echo_prompt else response_text
            sequence = truncate_and_tokenize_response_text(text, request, self.tokenizer, self.tokenizer_name)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
