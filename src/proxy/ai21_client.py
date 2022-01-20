import requests
from typing import Dict

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from .client import Client, wrap_request_time


class AI21Client(Client):
    """
    AI21 Labs provides Jurassic models.
    https://studio.ai21.com/docs/api/
    """

    def __init__(self, api_key: str, cache_path: str):
        self.api_key = api_key
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        model = request.model
        if model not in ["ai21/j1-large", "ai21/j1-jumbo"]:
            raise Exception("Invalid model")
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
            return requests.post(
                f"https://api.ai21.com/studio/v1/{request.model_engine}/complete",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

        cache_key = Client.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

        if "completions" not in response:
            return RequestResult(success=False, cached=False, error=response["detail"], completions=[])

        def fix_text(x: str, first: bool) -> str:
            x = x.replace("▁", " ")
            x = x.replace("<|newline|>", "\n")
            # For some reason, the first token sometimes starts with a space, so get rid of it
            if first and x.startswith(" "):
                x = x[1:]
            return x

        def parse_token(raw: Dict, first: bool) -> Token:
            return Token(
                text=fix_text(raw["generatedToken"]["token"], first),
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
