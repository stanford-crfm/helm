import openai
from typing import List

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from .client import Client, wrap_request_time


class OpenAIClient(Client):
    def __init__(self, api_key: str, cache_path: str):
        openai.api_key = api_key
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine,
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

        try:

            def do_it():
                return openai.Completion.create(**raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.error.InvalidRequestError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[])

        completions = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for text, logprob, top_logprobs in zip(
                raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
            ):
                # Do not include these excess tokens in the response.
                # TODO: this is a hacky solution until we figure out why
                #       OpenAI is sending tokens including and past the stop sequences.
                # TODO: This logic doesn't work when the stop sequences spans multiple tokens.
                #       https://github.com/stanford-crfm/benchmarking/issues/53
                if any(stop in text for stop in request.stop_sequences):
                    break

                # TODO: For some reason, the first log probability and top choices are None.
                #       https://github.com/stanford-crfm/benchmarking/issues/54
                tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                sequence_logprob += logprob or 0
            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completions.append(completion)
        return RequestResult(
            success=True, cached=cached, request_time=response["request_time"], completions=completions
        )
