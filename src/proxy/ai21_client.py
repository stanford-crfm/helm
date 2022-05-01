from typing import Dict, List
import requests

from dacite import from_dict

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken, TextRange
from .client import Client, wrap_request_time


class AI21RequestError(Exception):
    pass


class AI21Client(Client):
    """
    AI21 Labs provides Jurassic models.
    https://studio.ai21.com/docs/api/
    """

    def __init__(self, api_key: str, cache_path: str):
        self.api_key = api_key
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        model: str = request.model
        if model not in ["ai21/j1-large", "ai21/j1-jumbo", "ai21/j1-grande"]:
            raise ValueError(f"Invalid model: {model}")

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
            response = requests.post(
                f"https://api.ai21.com/studio/v1/{request.model_engine}/complete",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

            if "completions" not in response and "detail" in response:
                raise AI21RequestError(f"AI21 error: {response['detail']}")

            return response

        try:
            # We need to include the engine's name to differentiate among requests made for different model engines
            cache_key = Client.make_cache_key({"engine": request.model_engine, **raw_request}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except AI21RequestError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[])

        def fix_text(x: str, first: bool) -> str:
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
            return Token(
                # Text should not be longer than text_length. Since "▁" is always inserted
                # in the beginning, we truncate the text from the right.
                text=fix_text(raw["generatedToken"]["token"], first)[-text_length:] if text_length else "",
                logprob=raw["generatedToken"]["logprob"],
                top_logprobs=dict((fix_text(x["token"], first), x["logprob"]) for x in raw["topTokens"]),
            )

        def parse_sequence(raw: Dict, first: bool) -> Sequence:
            text = raw["text"]
            tokens = [parse_token(token, first and i == 0) for i, token in enumerate(raw["tokens"])]
            logprob = sum(token.logprob for token in tokens)
            return Sequence(text=text, logprob=logprob, tokens=tokens)

        prompt = parse_sequence(response["prompt"], True)
        completions = []
        for raw_completion in response["completions"]:
            completion = parse_sequence(raw_completion["data"], False)
            completions.append(prompt + completion if request.echo_prompt else completion)

        return RequestResult(
            success=True, cached=cached, request_time=response["request_time"], completions=completions
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

            if "tokens" not in response and "detail" in response:
                raise AI21RequestError(f"AI21 error when tokenizing: {response['detail']}")

            return response

        response, cached = self.cache.get(raw_request, do_it)

        # Each token is represented like this in the response:
        # {'token': '▁Hello', 'textRange': {'start': 0, 'end': 5}}
        tokens: List[TokenizationToken] = []
        for token_dict in response["tokens"]:
            tokens.append(
                TokenizationToken(text=token_dict["token"], text_range=from_dict(TextRange, token_dict["textRange"]))
            )
        text: str = response["text"]
        return TokenizationRequestResult(cached=cached, tokens=tokens, text=text)
