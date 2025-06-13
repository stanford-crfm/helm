from threading import Lock
from typing import Any, Dict, List, Optional, Union

import transformers

from helm.clients.client import CachingClient
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block
from helm.common.request import GeneratedOutput, Request, RequestResult, wrap_request_time
from helm.proxy.retry import NonRetriableException


_pipelines: Dict[str, transformers.Pipeline] = {}
_pipelines_lock: Lock = Lock()


def _get_pipeline(
    helm_model_name: str,
    pipeline_kwargs: Dict[str, Any],
) -> Any:
    """
    Checks if the desired HuggingFaceModel is cached. Creates the HuggingFaceModel if it's not cached.
    Returns the HuggingFaceModel.
    """
    global _pipelines
    global _pipelines_lock
    with _pipelines_lock:
        if helm_model_name not in _pipelines:
            huggingface_model_name = pipeline_kwargs["model"]
            with htrack_block(
                f"Loading HuggingFace model {huggingface_model_name} (kwargs={pipeline_kwargs}) "
                f"for HELM model {helm_model_name} with transformers.pipeline"
            ):
                _pipelines[helm_model_name] = transformers.pipeline(**pipeline_kwargs)

    return _pipelines[helm_model_name]


class HuggingFacePipelineClient(CachingClient):
    def __init__(
        self, cache_config: CacheConfig, model_name: str, pretrained_model_name_or_path: Optional[str] = None, **kwargs
    ):
        # Include `pretrained_model_name_or_path` parameter so that model deployments can use
        # the `pretrained_model_name_or_path` arg to override `model_name`
        super().__init__(cache_config=cache_config)
        self._helm_model_name = model_name
        self._pipeline_kwargs = {
            "model": pretrained_model_name_or_path or self._helm_model_name,
            "task": "text-generation",
            **kwargs,
        }
        self._pipeline = _get_pipeline(self._helm_model_name, self._pipeline_kwargs)

    def make_text_inputs(self, request: Request) -> Union[str, List[Dict[str, str]]]:
        if request.prompt and request.messages:
            raise NonRetriableException(f"More than one of `prompt` and `messages` was set in request: {request}")
        # Chat model expects a list of messages as input
        if self._pipeline.tokenizer.chat_template:
            if request.messages:
                return request.messages
            else:
                return [{"role": "user", "content": request.prompt}]
        # Base non-chat model expects a string as input
        else:
            if request.messages:
                raise
            else:
                return request.prompt

    def make_request(self, request: Request) -> RequestResult:
        """Make a request"""
        if request.model != self._helm_model_name:
            raise NonRetriableException(
                f"This instance of HuggingFacePipelineClient has loaded model {self._helm_model_name} but the request was for model {request.model}"  # noqa: E501
            )
        completions: List[GeneratedOutput] = []
        do_sample = request.temperature > 0.0
        raw_request = {
            "text_inputs": self.make_text_inputs(request),
            "return_full_text": request.echo_prompt,
            "temperature": request.temperature if do_sample else None,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k_per_token if do_sample else None,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
        }
        if request.stop_sequences:
            stop_sequence_ids = self._pipeline.tokenizer(
                request.stop_sequences, return_token_type_ids=False, add_special_tokens=False
            )
            if len(stop_sequence_ids.input_ids) == 1 and len(stop_sequence_ids.input_ids[0]) == 1:
                raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            else:
                raise NonRetriableException(
                    "Multiple stop sequences and stop sequences of multiple tokens, are not yet supported by HuggingFacePipelineClient"  # noqa: E501
                )

        def do_it() -> Dict[str, Any]:
            pipeline_outputs = self._pipeline(**raw_request)
            return {"outputs": pipeline_outputs}

        cache_key = CachingClient.make_cache_key(
            {"pipeline_kwargs": self._pipeline_kwargs, **raw_request},
            request,
        )

        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        for raw_output in response["outputs"]:
            completions.append(GeneratedOutput(text=raw_output["generated_text"], logprob=0, tokens=[]))
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )
