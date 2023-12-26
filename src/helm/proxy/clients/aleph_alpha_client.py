from typing import Any, Dict, List, Optional

from aleph_alpha_client import Client, CompletionRequest, CompletionResponse, Prompt

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from .client import CachingClient, truncate_sequence


class AlephAlphaClient(CachingClient):
    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._api_key: str = api_key
        self._aleph_alpha_client: Optional[Client] = None

    def _send_request(self, model: str, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if self._aleph_alpha_client is None:
            self._aleph_alpha_client = Client(token=self._api_key)

        request = CompletionRequest(prompt=Prompt.from_text(prompt), **parameters)
        response: CompletionResponse = self._aleph_alpha_client.complete(request, model=model)
        return dict(response.to_json())

    def make_request(self, request: Request) -> RequestResult:
        """Make a request following https://docs.aleph-alpha.com/api/complete."""
        model: str = request.model_engine
        prompt: str = request.prompt
        parameters = {
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
                result = self._send_request(model, prompt, parameters)
                assert "completions" in result, f"Invalid response: {result}"
                return result

            cache_key: Dict = CachingClient.make_cache_key({"model": model, "prompt": prompt, **parameters}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"AlephAlphaClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = []
        for completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            # `completion_tokens` is the list of selected tokens.
            for i, token in enumerate(completion.get("completion_tokens", [])):
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
