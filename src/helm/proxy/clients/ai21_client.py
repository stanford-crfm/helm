from typing import Dict, List
import requests

from dacite import from_dict

from helm.common.cache import Cache, CacheConfig
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
    TextRange,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time, truncate_sequence


class AI21RequestError(Exception):
    pass


class AI21Client(Client):
    """
    AI21 Labs provides Jurassic models.
    https://studio.ai21.com/docs/api/
    """

    COMPLETION_URL_TEMPLATE: str = "https://api.ai21.com/studio/v1/{model}/complete"
    EXPERIMENTAL_COMPLETION_URL_TEMPLATE: str = "https://api.ai21.com/studio/v1/experimental/{model}/complete"

    @staticmethod
    def handle_failed_request(api_type: str, response: Dict):
        error_message: str = f"AI21 {api_type} API error -"

        # Error messages are returned via 'detail' or 'Error' in response
        if "detail" in response:
            error_message += f" Detail: {response['detail']}"
        if "Error" in response:
            error_message += f" Error: {response['Error']}"

        raise AI21RequestError(error_message)

    def __init__(self, api_key: str, cache_config: CacheConfig):
        self.api_key = api_key
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {
            "prompt": request.prompt,
            "temperature": request.temperature,
            "numResults": request.num_completions,
            "topKReturn": request.top_k_per_token,
            "topP": request.top_p,
            "maxTokens": request.max_tokens,
            "stopSequences": request.stop_sequences,
        }

        def do_it():
            url_template: str = (
                AI21Client.EXPERIMENTAL_COMPLETION_URL_TEMPLATE
                if request.model_engine == "j1-grande-v2-beta"
                else AI21Client.COMPLETION_URL_TEMPLATE
            )
            response = requests.post(
                url_template.format(model=request.model_engine),
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

            # # If 'completions' is not present in the response, assume request failed.
            if "completions" not in response:
                AI21Client.handle_failed_request(api_type="completion", response=response)

            return response

        try:
            # We need to include the engine's name to differentiate among requests made for different model engines
            cache_key = Client.make_cache_key({"engine": request.model_engine, **raw_request}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except AI21RequestError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        def fix_text(x: str, first: bool) -> str:
            # TODO(#1522): check if with #1519 this is still needed. This is similar to #1516.
            x = x.replace("▁", " ")
            x = x.replace("<|newline|>", "\n")
            # For some reason, the first token sometimes starts with a space, so get rid of it
            if first and x.startswith(" "):
                x = x[1:]
            return x

        def parse_token(raw: Dict, first: bool) -> Token:
            """
            Parses a raw response token to a Token object.

            Sometimes a "▁" with length 0 is added to the beginning of a sequence
            or token by the AI21 tokenizer probably to mark the start of a new sequence.
            e.g. " burying him" -> ["▁"(0,0), "▁burying"(0,8), "▁him"(8,12)];
            "df\n---" -> '[▁df'(0,2), '\n'(2, 3), '▁---'(3, 6)]

            By computing the actual length of a token and truncating it from the right,
            We can remove those "▁"s so that the tokenization result aligns with the
            input prompt.
            """

            # Compute the actual length of the token text
            # e.g. "▁burying"(0,8) -> 8 - 0 = 8; "▁burying"(0,7) -> 7 - 0 = 7
            text_length: int = raw["textRange"]["end"] - raw["textRange"]["start"]
            # "topTokens" can be None when sending a request with topKReturn=0
            # AI21 sends unscaled logprobs as `raw_logprob` so use this instead of `logprob`.
            top_logprobs: Dict[str, float] = dict(
                (fix_text(x["token"], first), x["raw_logprob"]) for x in raw["topTokens"] or []
            )

            return Token(
                # Text should not be longer than text_length. Since "▁" is always inserted
                # in the beginning, we truncate the text from the right.
                text=fix_text(raw["generatedToken"]["token"], first)[-text_length:] if text_length else "",
                logprob=raw["generatedToken"]["raw_logprob"],
                top_logprobs=top_logprobs,
            )

        def parse_sequence(raw: Dict, first: bool, finish_reason: Dict = None) -> Sequence:
            text = raw["text"]
            tokens = [parse_token(token, first and i == 0) for i, token in enumerate(raw["tokens"])]
            logprob = sum(token.logprob for token in tokens)
            return Sequence(text=text, logprob=logprob, tokens=tokens, finish_reason=finish_reason)

        prompt = parse_sequence(response["prompt"], True)
        completions = []
        for raw_completion in response["completions"]:
            completion = parse_sequence(raw_completion["data"], False, raw_completion["finishReason"])
            completion = truncate_sequence(completion, request)
            completions.append(prompt + completion if request.echo_prompt else completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """
        Tokenizes the text by using the AI21 endpoint: https://api.ai21.com/studio/v1/tokenize.
        """
        raw_request: Dict[str, str] = {"text": request.text}

        def do_it():
            response = requests.post(
                "https://api.ai21.com/studio/v1/tokenize",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

            # If 'tokens' is not present in the response, assume request failed.
            if "tokens" not in response:
                AI21Client.handle_failed_request(api_type="tokenizer", response=response)

            return response

        try:
            response, cached = self.cache.get(raw_request, do_it)
        except AI21RequestError:
            return TokenizationRequestResult(success=False, cached=False, text="", tokens=[])

        # Each token is represented like this in the response:
        # {'token': '▁Hello', 'textRange': {'start': 0, 'end': 5}}
        tokens: List[TokenizationToken] = []
        for token_dict in response["tokens"]:
            tokens.append(
                TokenizationToken(value=token_dict["token"], text_range=from_dict(TextRange, token_dict["textRange"]))
            )
        text: str = response["text"]
        return TokenizationRequestResult(success=True, cached=cached, tokens=tokens, text=text)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Not supported")
