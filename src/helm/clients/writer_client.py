from typing import Any, Dict, List, Mapping, Optional

from helm.clients.client import CachingClient
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token

try:
    from writerai import Writer
    from writerai.types.chat_completion import ChatCompletion
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class WriterClient(CachingClient):
    def __init__(self, cache_config: CacheConfig, api_key: Optional[str] = None):
        super().__init__(cache_config=cache_config)
        self._writer_client = Writer(api_key=api_key)

    def _get_messages_from_request(self, request: Request) -> List[Dict]:
        if request.prompt and request.messages:
            raise ValueError(f"Only one of `prompt` and `messages` may be set in request: {request}")
        if request.multimodal_prompt:
            raise ValueError("`multimodal_prompt` is not supported by WriterClient")
        if request.messages:
            return [{"role": message["role"], "content": message["content"]} for message in request.messages]
        else:
            return [{"role": "user", "content": request.prompt}]

    def _convert_chat_completion_to_generated_outputs(
        self, chat_completion: ChatCompletion, request: Request
    ) -> List[GeneratedOutput]:
        generated_outputs: List[GeneratedOutput] = []
        for choice in chat_completion.choices:
            raw_completion_content = choice.message.content
            # The Writer chat completion API doesn't support echo.
            # If `echo_prompt` is true, combine the prompt and completion.
            text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content
            tokens: List[Token] = []
            if choice.logprobs and choice.logprobs.content:
                tokens = [
                    Token(text=choice_token.token, logprob=choice_token.logprob)
                    for choice_token in choice.logprobs.content
                ]
            generated_output = GeneratedOutput(
                text=text,
                logprob=sum(token.logprob for token in tokens) if tokens else 0.0,
                tokens=tokens,
                finish_reason={"reason": choice.finish_reason},
            )
            generated_outputs.append(generated_output)
        return generated_outputs

    def _convert_request_to_raw_request(self, request: Request) -> Dict:
        raw_request = {
            "messages": self._get_messages_from_request(request),
            "model": request.model.split("/")[-1],
            "logprobs": bool(request.top_k_per_token),
            "max_tokens": request.max_tokens,
            "n": request.num_completions,
            "stop": request.stop_sequences,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.response_format and request.response_format.json_schema:
            raw_request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": request.response_format.json_schema,
                },
            }
        return raw_request

    def make_request(self, request: Request) -> RequestResult:
        raw_request = self._convert_request_to_raw_request(request)
        cache_key: Mapping = CachingClient.make_cache_key(raw_request, request)

        def do_it() -> Dict[Any, Any]:
            return self._writer_client.chat.chat(**raw_request).model_dump()

        try:
            raw_response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            chat_completion: ChatCompletion = ChatCompletion.model_validate(raw_response)
        except Exception as error:
            return RequestResult(
                success=False,
                cached=False,
                error=str(error),
                completions=[],
                embedding=[],
            )

        generated_outputs = self._convert_chat_completion_to_generated_outputs(chat_completion, request)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=raw_response["request_time"],
            request_datetime=raw_response["request_datetime"],
            completions=generated_outputs,
            embedding=[],
        )
