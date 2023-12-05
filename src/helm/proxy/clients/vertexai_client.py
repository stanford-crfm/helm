# mypy: check_untyped_defs = False
import json
import requests
from typing import Any, Dict, List

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token, ErrorFlags
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient, truncate_sequence

try:
    import vertexai
    from vertexai.language_models import TextGenerationModel
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["vertexai"])
from toolbox.printing import debug, print_visible


class VertexAIClient(CachingClient):
    def __init__(self, tokenizer: Tokenizer, cache_config: CacheConfig, project_id: str, location: str) -> None:
        super().__init__(cache_config=cache_config)
        print_visible(f"VertexAIClient: project_id = {project_id}")
        self.project_id = project_id
        self.location = location
        self.tokenizer = tokenizer

        vertexai.init(project=self.project_id, location=self.location)

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        parameters = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "top_k": request.top_k_per_token,
            "top_p": request.top_p,
            # "stop_sequence": request.stop_sequences,
            # "candidate_count": request.num_completions,
            # "frequency_penalty": request.frequency_penalty,
            # "presence_penalty": request.presence_penalty,
            # "echo": request.echo_prompt,
        }

        completions: List[Sequence] = []
        model_name: str = request.model_engine

        try:

            def do_it():
                model = TextGenerationModel.from_pretrained(model_name)
                response = model.predict(request.prompt, **parameters)
                response_dict = {
                    "predictions": [{"text": completion.text for completion in response.candidates}],
                }  # TODO: Extract more information from the response
                return response_dict

            # We need to include the engine's name to differentiate among requests made for different model
            # engines since the engine name is not included in the request itself.
            # In addition, we want to make `request.num_completions` fresh
            # requests, cache key should contain the completion_index.
            # Echoing the original prompt is not officially supported by Writer. We instead prepend the
            # completion with the prompt when `echo_prompt` is true, so keep track of it in the cache key.
            cache_key = CachingClient.make_cache_key(
                {
                    "engine": request.model_engine,
                    "prompt": request.prompt,
                    **parameters,
                },
                request,
            )

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except (requests.exceptions.RequestException, AssertionError) as e:
            error: str = f"VertexAIClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        for prediction in response["predictions"]:
            response_text = prediction["text"]

            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                # Writer uses the GPT-2 tokenizer
                TokenizationRequest(request.prompt, tokenizer="huggingface/gpt2")
            )

            # Log probs are not currently not supported by the Writer, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=str(text), logprob=0, top_logprobs={}) for text in tokenization_result.raw_tokens
            ]

            completion = Sequence(text=response_text, logprob=0, tokens=tokens)
            sequence = truncate_sequence(completion, request, print_warning=True)
            completions.append(sequence)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
