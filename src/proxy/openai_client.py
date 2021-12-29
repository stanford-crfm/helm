import openai

from .client import Client
from common.cache import Cache
from common.request import Request, RequestResult, Completion


class OpenAIClient(Client):
    def __init__(self, api_key: str, cache_path: str):
        openai.api_key = api_key
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine(),
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.numSamples,
            "max_tokens": request.maxTokens,
            "best_of": request.topK,
            "stop": request.stopSequences or None,  # API doesn't like empty list
            "top_p": request.topP,
            "presence_penalty": request.presencePenalty,
            "frequency_penalty": request.frequencyPenalty,
            "logprobs": 1,
        }

        try:

            def do_it():
                return openai.Completion.create(**raw_request)

            response, cached = self.cache.get(raw_request, self.wrap_request_time(do_it))
        except openai.error.InvalidRequestError as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[])

        completions = []
        for choice in response["choices"]:
            completions.append(Completion(text=choice["text"]))
        return RequestResult(success=True, cached=cached, requestTime=response["requestTime"], completions=completions)
