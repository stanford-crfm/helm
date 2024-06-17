from copy import deepcopy
from itertools import zip_longest
import threading
from typing import List, Dict, Any, Mapping, Optional, TypedDict, Union

import requests
from retrying import retry

from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.clients.client import CachingClient, truncate_sequence, cleanup_str

try:
    from together import Together
    from together.types import ChatCompletionResponse, CompletionResponse
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["together"])


class _RewriteRequestTags:
    """Tags that indicate that the request for the model must be rewritten before sending to Together."""

    # TODO: Convert to StrEnum after upgrading to Python 3.11

    ADD_EOS_TOKEN_AS_STOP_SEQUENCE = "ADD_EOS_TOKEN_AS_STOP_SEQUENCE"
    """Indicates that the EOS token should be added as an extra stop sequence.

    This prevents the model from incorrectly returning the EOS token as part of the generation."""

    SET_DETAILS_TO_TRUE = "SET_DETAILS_TO_TRUE"
    """Indicates that the `details` field should be set to `true`.

    This indicates that Together should return logprobs for models that do not return logprobs by default."""


_MODEL_TO_TAGS: Dict[str, List[str]] = {
    "alpaca-7b": [_RewriteRequestTags.ADD_EOS_TOKEN_AS_STOP_SEQUENCE],
    "vicuna-7b-v1.3": [_RewriteRequestTags.ADD_EOS_TOKEN_AS_STOP_SEQUENCE],
    "llama-65b": [_RewriteRequestTags.SET_DETAILS_TO_TRUE],
    "llama-2-70b": [_RewriteRequestTags.SET_DETAILS_TO_TRUE],
    "vicuna-13b-v1.3": [_RewriteRequestTags.ADD_EOS_TOKEN_AS_STOP_SEQUENCE],
}
"""Dict of models to Together model tags.

This indicates which models require their requests to be rewritten before sending to together.
The keys are the model engine of the HELM model name (e.g. "alpaca-7b"), not the full HELM model name
(e.g. "stanford/alpaca-7b") or the Together model name (e.g. "togethercomputer/alpaca-7b")."""


_MODEL_TO_EOS_TOKEN: Dict[str, str] = {
    "alpaca-7b": "</s>",
    "vicuna-7b-v1.3": "</s>",
    "vicuna-13b-v1.3": "</s>",
}
"""Dict of models to end of sequence tokens.

This provides the end of sequence tokens for models that have `ADD_EOS_TOKEN_AS_STOP_SEQUENCE` as a model tag.
We hardcode the end of sequence tokens as constants here instead of attepmting to auto-infer them, for simplicity.
The keys are the model engine of the HELM model name (e.g. "alpaca-7b"), not the full HELM model name
(e.g. "stanford/alpaca-7b") or the Together model name (e.g. "togethercomputer/alpaca-7b")."""


TOGETHER_SUPPORTS_ASYNC_REQUESTS = False
"""Whether Together AI currently supports asynchronous requests."""


def _rewrite_raw_request_for_model_tags(raw_request: Dict[str, Any], model_engine: str) -> Dict[str, Any]:
    """Rewrite the raw request given the model."""
    # Make a deepcopy to avoid mutating the input in unexpected ways
    # (e.g. raw_request["stop"] can be a mutable list)
    rewritten_request = deepcopy(raw_request)
    model_tags = _MODEL_TO_TAGS.get(model_engine, [])
    for model_tag in model_tags:
        if model_tag == _RewriteRequestTags.ADD_EOS_TOKEN_AS_STOP_SEQUENCE:
            eos_token = _MODEL_TO_EOS_TOKEN.get(model_engine)
            if not eos_token:
                raise ValueError(f"Unknown EOS token for: {model_engine}")
            if isinstance(rewritten_request["stop"], list):
                rewritten_request["stop"].append(eos_token)
            else:
                rewritten_request["stop"] = [eos_token]
        elif model_tag == _RewriteRequestTags.SET_DETAILS_TO_TRUE:
            rewritten_request["details"] = True
        else:
            raise ValueError(f"Unknown `_RewriteRequestTags`: {model_tag}")
    return rewritten_request


class TogetherClientError(Exception):
    pass


class JobNotFinishedError(TogetherClientError):
    """Exception raised when trying to get a response for a Together async job that has not finished"""

    pass


class TogetherClient(CachingClient):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    INFERENCE_ENDPOINT: str = "https://api.together.xyz/api/inference"
    RETRIEVE_JOB_MAX_WAIT_SECONDS: int = 60

    def convert_to_raw_request(self, request: Request) -> Dict:
        # Following the examples from https://github.com/togethercomputer/open-models-api
        raw_request = {
            "request_type": "language-model-inference",
            "model": self.together_model or request.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
        }
        return _rewrite_raw_request_for_model_tags(raw_request, request.model_engine)

    def __init__(self, cache_config: CacheConfig, together_model: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(cache_config=cache_config)
        # TODO: the endpoint currently doesn't require an API key. When an API key is not specified
        #       in credentials.conf, we rely on offline evaluation only.
        self.api_key: Optional[str] = api_key
        self.together_model = together_model

    def _get_job_url(self, job_id: str) -> str:
        return f"https://api.together.xyz/jobs/job/{job_id}"

    def make_request(self, request: Request) -> RequestResult:
        raw_request = self.convert_to_raw_request(request)
        cache_key = CachingClient.make_cache_key(raw_request, request)

        if not self.api_key:
            raise TogetherClientError("togetherApiKey not set in credentials.conf")
        headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}

        if TOGETHER_SUPPORTS_ASYNC_REQUESTS:

            def submit_job() -> str:
                submit_request = {**raw_request, "async": True}
                submit_response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=submit_request)
                try:
                    submit_response.raise_for_status()
                except Exception as e:
                    raise TogetherClientError(
                        f"Together job submission request failed with {submit_response.status_code}: "
                        f"{submit_response.text}"
                    ) from e
                submit_response_json = submit_response.json()
                job_id = submit_response_json.get("id")
                if not job_id:
                    raise TogetherClientError(
                        f"Could not get job_id from job submission response {submit_response_json}"
                    )
                return job_id

            def retry_if_job_not_finished(exception: Exception) -> bool:
                return isinstance(exception, JobNotFinishedError)

            # Retry with a 5 second delay that increases by 5 seconds each attempt with a maximum delay of 30 seconds.
            # Stop retrying after 5 minutes.
            @retry(
                retry_on_exception=retry_if_job_not_finished,
                wait_incrementing_start=5 * 1000,  # 5 seconds
                wait_incrementing_increment=5 * 1000,  # 5 seconds
                wait_incrementing_max=30 * 1000,  # 30 seconds
                stop_max_delay=5 * 60 * 1000,  # 5 minutes
            )
            def retrieve_job(job_id: str) -> Dict[Any, Any]:
                job_url = self._get_job_url(job_id)
                retrieve_response = requests.get(job_url, headers=headers)
                try:
                    retrieve_response.raise_for_status()
                except Exception as e:
                    raise TogetherClientError(
                        f"Together job retrieval request failed with {retrieve_response.status_code}: "
                        f"{retrieve_response.text}"
                    ) from e
                retrieve_response_json = retrieve_response.json()
                if retrieve_response_json["status"] != "finished":
                    raise JobNotFinishedError(f"Together job not finished: {job_id}")
                if "output" not in retrieve_response_json:
                    raise TogetherClientError(
                        f"Could not get output from Together job {job_id}: {retrieve_response_json}"
                    )
                if "error" in retrieve_response_json["output"]:
                    error_message = retrieve_response_json["output"]["error"]
                    raise TogetherClientError(f"Together request (job_id={job_id}) failed with error: {error_message}")
                return retrieve_response_json["output"]

            def do_it_async() -> Dict[Any, Any]:
                job_id = submit_job()
                return retrieve_job(job_id)

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it_async))
        else:

            def do_it_sync() -> Dict[Any, Any]:
                response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request)
                try:
                    response.raise_for_status()
                except Exception as e:
                    raise TogetherClientError(
                        f"Together request failed with {response.status_code}: {response.text}"
                    ) from e
                result = response.json()
                if "output" not in result:
                    raise TogetherClientError(f"Could not get output from Together response: {result}")
                if "error" in result["output"]:
                    error_message = result["output"]["error"]
                    raise TogetherClientError(f"Together request failed with error: {error_message}")
                return result["output"]

            try:
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it_sync))
            except Exception as error:
                return RequestResult(
                    success=False,
                    cached=False,
                    error=str(error),
                    completions=[],
                    embedding=[],
                )

        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            # TODO: take this out when "logprobs" is supported properly in batch/offline mode
            # Currently, token_logprobs is provided in interactive/online mode but it has a different format
            # Waiting for a fix.
            if "logprobs" in raw_completion:
                raw_data = raw_completion["logprobs"]
                for text, logprob in zip(raw_data["tokens"], raw_data["token_logprobs"]):
                    # TODO #1654: Check if this is still needed
                    text = cleanup_str(text, "together")
                    tokens.append(Token(text=text, logprob=logprob or 0))
                    sequence_logprob += logprob or 0
            else:
                # hack: just make the entire text one token so that something shows up in the frontend
                text = cleanup_str(raw_completion["text"], "together")
                tokens.append(Token(text=text, logprob=0))

            raw_finish_reason: Optional[str] = raw_completion.get("finish_reason")
            finish_reason: Optional[Dict] = {"reason": raw_finish_reason} if raw_finish_reason else None

            completion = GeneratedOutput(
                text=cleanup_str(raw_completion["text"], "together"),
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason=finish_reason,
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        request_time: Union[float, Dict[str, Any]] = response["request_time"]
        if isinstance(request_time, dict):
            batch_performance_metadata: Dict = response["request_time"]
            return RequestResult(
                success=True,
                cached=cached,
                request_time=0,
                completions=completions,
                batch_size=batch_performance_metadata["batch_size"],
                batch_request_time=batch_performance_metadata["batch_time"],
                embedding=[],
            )
        else:
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response["raw_compute_time"] if "raw_compute_time" in response else request_time,
                completions=completions,
                embedding=[],
            )


_MODEL_TO_DEFAULT_STOP_TOKENS: Optional[Mapping[str, List[str]]] = None
_MODEL_TO_DEFAULT_STOP_TOKENS_LOCK = threading.Lock()


def get_default_stop_tokens_for_model(together_model: str, together_client: Together) -> List[str]:
    global _MODEL_TO_DEFAULT_STOP_TOKENS
    global _MODEL_TO_DEFAULT_STOP_TOKENS_LOCK
    with _MODEL_TO_DEFAULT_STOP_TOKENS_LOCK:
        if _MODEL_TO_DEFAULT_STOP_TOKENS is None:
            _MODEL_TO_DEFAULT_STOP_TOKENS = {}
            for model in together_client.models.list():
                _MODEL_TO_DEFAULT_STOP_TOKENS[model.id.lower()] = model.config["stop"]
    stop_tokens = _MODEL_TO_DEFAULT_STOP_TOKENS.get(together_model.lower())
    if stop_tokens is None:
        raise ValueError(f"Unknown together_model {together_model}")
    return stop_tokens


class TogetherRawChatRequest(TypedDict):
    messages: List[Dict[str, str]]
    model: str
    max_tokens: int
    stop: List[str]
    temperature: float
    top_p: float
    top_k: int
    logprobs: int
    echo: bool
    n: int


class TogetherChatClient(CachingClient):
    """Client that uses the Python Together library for chat models."""

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str], together_model: Optional[str] = None):
        super().__init__(cache_config=cache_config)
        self._client = Together(api_key=api_key)
        self._together_model = together_model

    def convert_to_raw_chat_request(self, request: Request) -> TogetherRawChatRequest:
        if request.messages:
            messages = request.messages
        else:
            messages = [{"role": "user", "content": request.prompt}]
        if self._together_model is not None:
            model = self._together_model
        else:
            model = request.model
        return {
            "messages": messages,
            "model": model,
            "max_tokens": request.max_tokens,
            "stop": request.stop_sequences + get_default_stop_tokens_for_model(model, self._client),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k_per_token,
            "logprobs": min(request.top_k_per_token, 1),
            "echo": request.echo_prompt,
            "n": request.num_completions,
        }

    def make_request(self, request: Request) -> RequestResult:
        raw_request = self.convert_to_raw_chat_request(request)
        cache_key = CachingClient.make_cache_key(raw_request, request)

        def do_it() -> Dict[Any, Any]:
            response = self._client.chat.completions.create(**raw_request)
            return response.model_dump(mode="json")

        try:
            raw_response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            response = ChatCompletionResponse.model_validate(raw_response)
        except Exception as error:
            return RequestResult(
                success=False,
                cached=False,
                error=str(error),
                completions=[],
                embedding=[],
            )

        generated_outputs: List[GeneratedOutput] = []
        for choice in response.choices:
            # NOTE: Together always returns None for choice.finish_reason
            # NOTE: Together does not return logprobs for the whole generated output, only for individual tokens
            tokens: List[Token] = []
            if choice.logprobs:
                for token_text, token_logprob in zip_longest(
                    choice.logprobs.tokens or [], choice.logprobs.token_logprobs or []
                ):
                    if token_text is None:
                        break
                    tokens.append(Token(text=token_text, logprob=token_logprob or 0.0))
            assert choice.message.role == "assistant"
            generated_outputs.append(GeneratedOutput(text=choice.message.content, logprob=0.0, tokens=tokens))
        return RequestResult(
            success=True,
            cached=cached,
            request_time=raw_response["request_time"],
            request_datetime=raw_response["request_datetime"],
            completions=generated_outputs,
            embedding=[],
        )


class TogetherRawCompletionRequest(TypedDict):
    prompt: str
    model: str
    max_tokens: int
    stop: List[str]
    temperature: float
    top_p: float
    top_k: int
    logprobs: int
    echo: bool
    n: int


class TogetherCompletionClient(CachingClient):
    """Client that uses the Python Together library for text completion models."""

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str], together_model: Optional[str] = None):
        super().__init__(cache_config=cache_config)
        self._client = Together(api_key=api_key)
        self._together_model = together_model

    def convert_to_raw_completion_request(self, request: Request) -> TogetherRawCompletionRequest:
        if self._together_model is not None:
            model = self._together_model
        else:
            model = request.model
        return {
            "prompt": request.prompt,
            "model": model,
            "max_tokens": request.max_tokens,
            "stop": request.stop_sequences + get_default_stop_tokens_for_model(model, self._client),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k_per_token,
            "logprobs": min(request.top_k_per_token, 1),
            "echo": request.echo_prompt,
            "n": request.num_completions,
        }

    def make_request(self, request: Request) -> RequestResult:
        raw_request = self.convert_to_raw_completion_request(request)
        cache_key = CachingClient.make_cache_key(raw_request, request)

        def do_it() -> Dict[Any, Any]:
            response = self._client.completions.create(**raw_request)
            return response.model_dump(mode="json")

        try:
            raw_response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            response = CompletionResponse.model_validate(raw_response)
        except Exception as error:
            return RequestResult(
                success=False,
                cached=False,
                error=str(error),
                completions=[],
                embedding=[],
            )

        generated_outputs: List[GeneratedOutput] = []
        for choice in response.choices:
            # NOTE: Together always returns None for choice.finish_reason
            # NOTE: Together does not return logprobs for the whole generated output, only for individual tokens
            tokens: List[Token] = []
            if choice.logprobs:
                for token_text, token_logprob in zip_longest(
                    choice.logprobs.tokens or [], choice.logprobs.token_logprobs or []
                ):
                    if token_text is None:
                        break
                    tokens.append(Token(text=token_text, logprob=token_logprob or 0.0))
            assert choice.text
            generated_outputs.append(GeneratedOutput(text=choice.text, logprob=0.0, tokens=tokens))
        return RequestResult(
            success=True,
            cached=cached,
            request_time=raw_response["request_time"],
            request_datetime=raw_response["request_datetime"],
            completions=generated_outputs,
            embedding=[],
        )
