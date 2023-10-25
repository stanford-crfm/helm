import json
import requests
from typing import Any, Dict, List

from aleph_alpha_client import Client as AlephAlphaPythonClient

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_sequence


class AlephAlphaClient(CachingClient):
    COMPLETION_ENDPOINT: str = "complete"

    def __init__(self, api_key: str, tokenizer: Tokenizer, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config, tokenizer=tokenizer)
        self.api_key: str = api_key
        self._aleph_alpha_client = AlephAlphaPythonClient(token=api_key)

    def _send_request(self, endpoint: str, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.request(
            method="POST",
            url=f"https://api.aleph-alpha.com/{endpoint}",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(raw_request),
            # Setting the nice flag prevents intensive benchmarking runs from saturating Aleph Alpha's API queues
            params=json.dumps({"nice": True}),
        )
        result = json.loads(response.text)
        assert "error" not in result, f"Request failed with error: {result['error']}"
        return result

    def make_request(self, request: Request) -> RequestResult:
        """Make a request following https://docs.aleph-alpha.com/api/complete."""
        raw_request = {
            "model": request.model_engine,
            "prompt": request.prompt,
            "maximum_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k_per_token,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "n": request.num_completions,
            "stop_sequences": request.stop_sequences,
            "log_probs": request.top_k_per_token,
            "echo": request.echo_prompt,
            "tokens": True,  # Setting to True returns individual tokens of the completion
        }

        try:

            def do_it():
                result = self._send_request(AlephAlphaClient.COMPLETION_ENDPOINT, raw_request)
                assert "completions" in result, f"Invalid response: {result}"
                return result

            response, cached = self.cache.get(raw_request, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"AlephAlphaClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = []
        for completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            # `completion_tokens` is the list of selected tokens.
            for i, token in enumerate(completion["completion_tokens"]):
                # Get the top K logprobs for the ith token
                top_logprobs: Dict[str, float] = completion["log_probs"][i]
                # Use the selected token value to get the logprob
                logprob: float = top_logprobs[token]
                sequence_logprob += logprob
                tokens.append(
                    Token(
                        text=token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                    )
                )

            sequence: Sequence = Sequence(text=completion["completion"], logprob=sequence_logprob, tokens=tokens)
            sequence = truncate_sequence(sequence, request)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
