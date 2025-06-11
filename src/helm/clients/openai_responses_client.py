# mypy: check_untyped_defs = False
from typing import Any, Dict, List, Optional, cast, Union

from helm.clients.openai_client import OpenAIClientUtils
from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE
from helm.common.request import (
    Thinking,
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
)
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.clients.client import (
    CachingClient,
    truncate_sequence,
    generate_uid_for_multimodal_prompt,
)
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])

class OpenAIResponseClient(CachingClient):
    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        openai_model_name: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.client = OpenAI(api_key=api_key, organization=org_id, base_url=base_url)
        self.reasoning_effort = reasoning_effort
        self.openai_model_name = openai_model_name

    def _get_cache_key(self, raw_request: Dict, request: Request):
        cache_key = CachingClient.make_cache_key(raw_request, request)
        if request.multimodal_prompt:
            prompt_key: str = generate_uid_for_multimodal_prompt(
                request.multimodal_prompt
            )
            cache_key = {**cache_key, "multimodal_prompt": prompt_key}
        return cache_key
    
    def _make_raw_request(self, request: Request) -> dict[str, Any]:
        input: Union[str, List[Dict[str, Any]]]
        if request.multimodal_prompt is not None:
            content = []
            request.validate()
            for media_object in request.multimodal_prompt.media_objects:
                if media_object.is_type("image") and media_object.location:
                    from helm.common.images_utils import encode_base64

                    base64_image: str = encode_base64(media_object.location)
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    )
                elif media_object.is_type(TEXT_TYPE):
                    assert media_object.text is not None
                    content.append({"type": "input_text", "text": media_object.text})
                else:
                    raise ValueError(
                        f"Unrecognized MediaObject type {media_object.type}"
                    )
            input = [{"role": "user", "content": content}]
        else:
            input = request.prompt

        raw_request: Dict[str, Any] = {
            "model": self.openai_model_name or request.model_engine,
            "input": input,
            "top_p": request.top_p,
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
            # Don't store responses for later retrieval
            "store": False,
        }
        if self.reasoning_effort:
            raw_request["reasoning"] = {
                "effort": self.reasoning_effort
            }
        # If o-series model, get reasoning summaries
        # Plus other changes
        # TODO: Refactor with line 226 in openai_client.py to common util
        model_engine: str = request.model_engine
        if OpenAIClientUtils.is_reasoning_model(model_engine):
            raw_request["reasoning"]["summary"] = "detailed"
            # Avoid error:
            # "Error code: 400 - {'error': {'message': "Unsupported parameter: 'temperature' is
            # not supported with this model.", 'type': 'invalid_request_error', 'param': 'temperature',
            # 'code': 'unsupported_parameter'}}"
            raw_request.pop("temperature", None)

            # The following parameters also happen to be unsupported by the o-series (code unsupported_parameter)
            raw_request.pop("top_p", None)
        
        return raw_request

    def make_request(self, request: Request) -> RequestResult:
        # Content can either be text or a list of multimodal content made up of text and images:
        # https://platform.openai.com/docs/api-reference/responses/create
        raw_request = self._make_request(request)

        def do_it() -> Dict[str, Any]:
            raw_response = self.client.responses.create(**raw_request).model_dump(
                mode="json"
            )
            assert not raw_response["error"], f"Error in response: {raw_response}"
            return raw_response

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            return OpenAIClientUtils.handle_openai_error(e, request)

        completions: List[GeneratedOutput] = []

        # Look for the completiom with type: "message" for the output
        # (based on observing the API on 11.06.2025)
        message_output = list(filter(response["output"], lambda x: x["type"] == "message"))
        # Expect length of 1, else, OpenAI API has changed, and we don't know what to do!
        assert len(message_output) == 1, "Unexpected response format from OpenAI API - > 1 message response provided!"
        message_output = message_output[0]

        # Similarly, look for the reasoning output based on API observations
        reasoning_output = list(filter(response["output"], lambda x: x["type"] == "reasoning"))
        # Either we have no reasoning, or some reasoning
        assert len(reasoning_output) <= 1, "Unexpected response format from OpenAI API - > 1 reasoning response provided!"
        for output in response["output"]:
            output_type = output["type"] # one of "message" or "reasoning" from API observation
            is_reasoning_output = None
            output_key = None
            match output_type:
                case "message":
                    output_key = "content"
                    is_reasoning_output = False
                case "reasoning":
                    output_key = "summary"
                    is_reasoning_output = True

            for raw_completion in output[output_key]:
                raw_completion_content = raw_completion["text"]
                text: str = (
                    request.prompt + raw_completion_content
                    if request.echo_prompt and not is_reasoning_output
                    else raw_completion_content
                )
                tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                    TokenizationRequest(text, tokenizer=self.tokenizer_name)
                )
                tokens: List[Token] = [
                    Token(text=cast(str, raw_token), logprob=0)
                    for raw_token in tokenization_result.raw_tokens
                ]
                completion = None
                if is_reasoning_output:
                    # Openai provides thinking outputs separately to main completion output
                    # So we have to add it separately
                    completion = GeneratedOutput(
                        text="",
                        logprob=0,  # OpenAI does not provide logprobs
                        tokens=tokens,
                        thinking=Thinking(text=raw_completion_content),
                    )
                else:
                    completion = GeneratedOutput(
                        text=text,
                        logprob=0,  # OpenAI does not provide logprobs
                        tokens=tokens,
                    )
                completions.append(
                    truncate_sequence(completion, request)
                )  # Truncate the text by stop sequences
        
        # Construct reasoning if it exists

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )
