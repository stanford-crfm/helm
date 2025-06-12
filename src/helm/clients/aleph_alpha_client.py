from typing import List

from helm.common.cache import CacheConfig
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.clients.client import CachingClient, truncate_sequence, generate_uid_for_multimodal_prompt

try:
    from aleph_alpha_client import Client, CompletionRequest, CompletionResponse, Image, Prompt
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["aleph-alpha"])


class AlephAlphaClient(CachingClient):
    def __init__(self, api_key: str, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._api_key: str = api_key
        self._aleph_alpha_client = Client(token=self._api_key) if self._api_key else None

    def make_request(self, request: Request) -> RequestResult:
        """Make a request following https://docs.aleph-alpha.com/api/complete."""
        assert self._aleph_alpha_client is not None

        model: str = request.model_engine
        prompt: Prompt

        # The prompt key is a unique identifier for the prompt
        prompt_key: str = request.prompt

        # Contents can either be text or a list of multimodal content made up of text, images or other content
        if request.multimodal_prompt is not None:
            from helm.common.images_utils import encode_base64

            items = []
            for media_object in request.multimodal_prompt.media_objects:
                if media_object.is_type("image") and media_object.location:
                    items.append(Image(base_64=encode_base64(media_object.location), cropping=None, controls=[]))
                elif media_object.is_type(TEXT_TYPE):
                    if media_object.text is None:
                        raise ValueError("MediaObject of text type has missing text field value")
                    items.append(media_object.text)
                else:
                    raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

            prompt = Prompt(items=items)
            prompt_key = generate_uid_for_multimodal_prompt(request.multimodal_prompt)
        else:
            prompt = Prompt.from_text(request.prompt)

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
                assert self._aleph_alpha_client is not None
                completion_response: CompletionResponse = self._aleph_alpha_client.complete(
                    request=CompletionRequest(prompt=prompt, **parameters), model=model
                )
                result = dict(completion_response.to_json())
                assert "completions" in result, f"Invalid response: {result}"
                return result

            cache_key = CachingClient.make_cache_key({"model": model, "prompt": prompt_key, **parameters}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"AlephAlphaClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            # `completion_tokens` is the list of selected tokens.
            for i, token in enumerate(completion.get("completion_tokens", [])):
                # Use the selected token value to get the logprob
                logprob: float = completion["log_probs"][i][token]
                sequence_logprob += logprob
                tokens.append(
                    Token(
                        text=token,
                        logprob=logprob,
                    )
                )

            sequence: GeneratedOutput = GeneratedOutput(
                text=completion["completion"], logprob=sequence_logprob, tokens=tokens
            )
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
