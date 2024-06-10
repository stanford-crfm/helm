# mypy: check_untyped_defs = False
from dataclasses import replace
from typing import Any, Dict, List, Optional, cast, Union

from helm.benchmark.model_metadata_registry import is_vlm
from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_sequence, generate_uid_for_multimodal_prompt

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class OpenAIClient(CachingClient):
    END_OF_TEXT: str = "<|endoftext|>"

    # Error OpenAI throws when the image in the prompt violates their content policy
    INAPPROPRIATE_IMAGE_ERROR: str = "Your input image may contain content that is not allowed by our safety system"

    # Set the finish reason to this if the prompt violates OpenAI's content policy
    CONTENT_POLICY_VIOLATED_FINISH_REASON: str = (
        "The prompt violates OpenAI's content policy. "
        "See https://labs.openai.com/policies/content-policy for more information."
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.client = OpenAI(api_key=api_key, organization=org_id, base_url=base_url)

    def _is_chat_model_engine(self, model_engine: str) -> bool:
        if model_engine == "gpt-3.5-turbo-instruct":
            return False
        elif model_engine.startswith("gpt-3.5") or model_engine.startswith("gpt-4"):
            return True
        return False

    def _get_model_for_request(self, request: Request) -> str:
        return request.model_engine

    def _get_cache_key(self, raw_request: Dict, request: Request):
        cache_key = CachingClient.make_cache_key(raw_request, request)
        if request.multimodal_prompt:
            prompt_key: str = generate_uid_for_multimodal_prompt(request.multimodal_prompt)
            cache_key = {**cache_key, "multimodal_prompt": prompt_key}
            del cache_key["messages"]
        return cache_key

    def _make_embedding_request(self, request: Request) -> RequestResult:
        raw_request: Dict[str, Any]
        raw_request = {
            "input": request.prompt,
            # Note: In older deprecated versions of the OpenAI API, "model" used to be "engine".
            "model": self._get_model_for_request(request),
        }

        def do_it() -> Dict[str, Any]:
            return self.client.embeddings.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # If the user is requesting completions instead of an embedding, then `completions`
        # needs to be populated, and `embedding` should be an empty list and vice-versa.
        embedding: List[float] = []
        # If the user is requesting an embedding instead of completion
        # then completions would be left as an empty list. The embedding needs to be set.
        embedding = response["data"][0]["embedding"]

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[],
            embedding=embedding,
        )

    def _make_chat_request(self, request: Request) -> RequestResult:
        messages: Optional[List[Dict[str, Union[str, Any]]]] = request.messages
        if (
            (request.prompt and request.messages)
            or (request.prompt and request.multimodal_prompt)
            or (request.messages and request.multimodal_prompt)
        ):
            raise ValueError(
                f"More than one of `prompt`, `messages` and `multimodal_prompt` was set in request: {request}"
            )
        if request.messages is not None:
            # Checks that all messages have a role and some content
            for message in request.messages:
                if not message.get("role") or not message.get("content"):
                    raise ValueError("All messages must have a role and content")
            # Checks that the last role is "user"
            if request.messages[-1]["role"] != "user":
                raise ValueError("Last message must have role 'user'")
            if request.prompt != "":
                hlog("WARNING: Since message is set, prompt will be ignored")
        else:
            # Convert prompt into a single message
            # For now, put the whole prompt in a single user message, and expect the response
            # to be returned in a single assistant message.
            # TODO: Support ChatML for creating multiple messages with different roles.
            # See: https://github.com/openai/openai-python/blob/main/chatml.md

            # Content can either be text or a list of multimodal content made up of text and images:
            # https://platform.openai.com/docs/guides/vision
            content: Union[str, List[Union[str, Any]]]
            if request.multimodal_prompt is not None:
                content = []
                for media_object in request.multimodal_prompt.media_objects:
                    if media_object.is_type("image") and media_object.location:
                        from helm.common.images_utils import encode_base64

                        base64_image: str = encode_base64(media_object.location)
                        image_object: Dict[str, str] = {"url": f"data:image/jpeg;base64,{base64_image}"}
                        content.append({"type": "image_url", "image_url": image_object})
                    elif media_object.is_type(TEXT_TYPE):
                        if media_object.text is None:
                            raise ValueError("MediaObject of text type has missing text field value")
                        content.append({"type": media_object.type, "text": media_object.text})
                    else:
                        raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

            else:
                content = request.prompt

            messages = [{"role": "user", "content": content}]

        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            # Note: Chat models may require adding an extra token to max_tokens
            # for the internal special role token.
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        # OpenAI's vision API doesn't allow None values for stop.
        # Fails with "body -> stop: none is not an allowed value" if None is passed.
        if is_vlm(request.model) and raw_request["stop"] is None:
            raw_request.pop("stop")

        def do_it() -> Dict[str, Any]:
            return self.client.chat.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            if self.INAPPROPRIATE_IMAGE_ERROR in str(e):
                hlog(f"Failed safety check: {str(request)}")
                empty_completion = GeneratedOutput(
                    text="",
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
                )
                return RequestResult(
                    success=True,
                    cached=False,
                    request_time=0,
                    completions=[empty_completion] * request.num_completions,
                    embedding=[],
                )

            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            # The OpenAI chat completion API doesn't support echo.
            # If `echo_prompt` is true, combine the prompt and completion.
            raw_completion_content = raw_completion["message"]["content"]
            text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content
            # The OpenAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                TokenizationRequest(text, tokenizer=self.tokenizer_name)
            )
            # Log probs are not currently not supported by the OpenAI chat completion API, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=cast(str, raw_token), logprob=0) for raw_token in tokenization_result.raw_tokens
            ]
            completion = GeneratedOutput(
                text=text,
                logprob=0,  # OpenAI does not provide logprobs
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            completions.append(truncate_sequence(completion, request))  # Truncate the text by stop sequences

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def _to_raw_completion_request(self, request: Request) -> Dict[str, Any]:
        raw_request: Dict[str, Any] = {
            # Note: In older deprecated versions of the OpenAI API, "model" used to be "engine".
            "model": self._get_model_for_request(request),
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "echo": request.echo_prompt,
        }

        # OpenAI doesn't let you ask for more completions than the number of
        # per-token candidates.
        raw_request["best_of"] = max(raw_request["best_of"], raw_request["n"])
        raw_request["logprobs"] = max(raw_request["logprobs"], raw_request["n"])

        return raw_request

    def _make_completion_request(self, request: Request) -> RequestResult:
        raw_request = self._to_raw_completion_request(request)

        def do_it() -> Dict[str, Any]:
            return self.client.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for (
                text,
                logprob,
            ) in zip(raw_data["tokens"], raw_data["token_logprobs"]):
                tokens.append(Token(text=text, logprob=logprob or 0))
                sequence_logprob += logprob or 0
            completion = GeneratedOutput(
                text=raw_completion["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            # OpenAI sends us back tokens past the end of text token,
            # so we need to manually truncate the list of tokens.
            # TODO: filed an issue with their support to check what the expected behavior here is.
            completion = truncate_sequence(
                completion, replace(request, stop_sequences=request.stop_sequences + [OpenAIClient.END_OF_TEXT])
            )
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return self._make_embedding_request(request)
        elif self._is_chat_model_engine(request.model_engine):
            return self._make_chat_request(request)
        else:
            return self._make_completion_request(request)
