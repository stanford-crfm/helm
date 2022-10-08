from typing import List, Dict, Any, Union
import time
import requests
from dataclasses import replace

from common.cache import Cache, CacheConfig
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from common.hierarchical_logger import hlog
from .client import Client, wrap_request_time, truncate_sequence

NO_NEWLINE_MODELS = ["together/t5", "together/t0pp", "together/glm"]

def prepare_text(x: str, model: str) -> str:
    """Process text before sending to API."""
    if model in NO_NEWLINE_MODELS:
        x = x.replace("\n", "<br>")
    return x

def fix_text(x: str, model: str) -> str:
    """Fix text that comes back from the API."""
    if model in NO_NEWLINE_MODELS:
        x = x.replace("▁", " ")
        x = x.replace("</s>", "")
        x = x.replace("<br>", "\n")
    return x


class TogetherClient(Client):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    ORGANIZATION: str = "together"

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        # Uses the same parameter names as the OpenAI API: https://beta.openai.com/docs/api-reference/completions
        return {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
        }

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = TogetherClient.convert_to_raw_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)

        def tweak_request(raw_request: Dict[str, Any]) -> Dict[str, Any]:
            # TODO: unify with the logic in export_requests.py
            raw_request = dict(raw_request)
            raw_request["prompt"] = prepare_text(raw_request["prompt"], request.model)
            raw_request["prompt"] = [raw_request["prompt"]]  # In the interactive API only for now
            raw_request["model"] = raw_request.pop("engine")
            raw_request["request_type"] = "language-model-inference"
            return raw_request

        try:

            def do_it():
                # Submit job
                response = requests.post("https://planetd.shift.ml/jobs", json={
                    "type": "general",
                    "payload": tweak_request(raw_request),
                    "returned_payload": {},
                    "status": "submitted",
                    "source": "dalle",
                }).json()

                # Poll and wait for job to be finished
                job_id = response['id']
                for t in range(10000000):
                    response = requests.get(f"https://planetd.shift.ml/job/{job_id}").json()
                    status = response["status"]
                    hlog(f"TogetherClient: Waiting for job {job_id}, status is {status}, waited {t} seconds")
                    if status == 'finished':
                        return response["returned_payload"]["result"]
                    elif status == 'failed':
                        raise Exception("TogetherClient request failed: {json.dumps(result)}")
                    time.sleep(1)


            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"TogetherClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[Sequence] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            # TODO: take this out when "logprobs" is supported
            if "logprobs" in raw_completion:
                raw_data = raw_completion["logprobs"]
                for text, logprob, top_logprobs in zip(
                    raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
                ):
                    text = fix_text(text, request.model)
                    tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                    sequence_logprob += logprob or 0
            else:
                # hack: just make the entire text one token so that something shows up in the frontend
                text = fix_text(raw_completion["text"], request.model)
                tokens.append(Token(text=text, logprob=0, top_logprobs={}))

            completion = Sequence(
                text=fix_text(raw_completion["text"], request.model),
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        print(response)
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
                request_time=request_time,
                completions=completions,
                embedding=[],
            )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
