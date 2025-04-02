from abc import ABC
from abc import abstractmethod

from helm.common.hierarchical_logger import hlog
from helm.common.cache import CacheConfig
from helm.common.request import (
    Request,
    RequestResult,
    Token,
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    GeneratedOutput,
)

from helm.clients.client import CachingClient
from helm.common.optional_dependencies import handle_module_not_found_error
from typing import TypeVar, Generic
from typing import Any, Dict, List
from threading import Semaphore, Lock
import threading

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.schema import (
        TextChatParameters,
        TextGenParameters,
        ReturnOptionProperties,
    )

except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["ibm"])

# Define the maximum number of parallel executions is limited by IBM API
MAX_CONCURRENT_REQUESTS = 8
__semaphores: Dict[str, Semaphore] = dict()
__semaphores_lock = Lock()


def _get_semaphore(model: str) -> Semaphore:
    with __semaphores_lock:
        if model not in __semaphores:
            __semaphores[model] = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

    return __semaphores[model]


T = TypeVar("T", TextGenParameters, TextChatParameters)


class ModelInferenceHandler(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, inference_engine: ModelInference):
        """
        :type inference_engine: object
        """
        self.inference_engine = inference_engine

    @abstractmethod
    def serve_request(self, prompt: str, params: T) -> Dict:
        pass

    @abstractmethod
    def parse_response(self, response: dict) -> List[GeneratedOutput]:
        pass

    @abstractmethod
    def create_params(self, request: Request) -> T:
        pass


class GenerateInferenceHandler(ModelInferenceHandler[TextGenParameters]):

    def __init__(self, inference_engine: ModelInference):
        self.inference_engine = inference_engine

    def create_params(self, request: Request) -> TextGenParameters:
        def set_temperature_requirements():
            # Default temperature 0.05 required by ibm/granite-13b-instruct-v2
            if self.inference_engine.model_id == "ibm/granite-13b-instruct-v2":
                return 0.05
            return 1e-7 if request.temperature == 0 else request.temperature

        return TextGenParameters(
            temperature=set_temperature_requirements(),
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            return_options=ReturnOptionProperties(
                input_text=True,
                generated_tokens=True,
                input_tokens=False,
                token_logprobs=True,
                token_ranks=False,
            ),
            include_stop_sequence=False,
            prompt_variables=None,
        )

    def serve_request(self, prompt: str, params: TextGenParameters) -> Dict:
        semaphore = _get_semaphore(self.inference_engine.model_id)

        with semaphore:
            response = self.inference_engine.generate(
                prompt=prompt,
                params=params,
            )
        return response

    def parse_response(self, response: dict) -> List[GeneratedOutput]:
        completions = []
        try:
            for r in response["results"]:
                sequence_logprob: float = 0
                tokens: List[Token] = []
                generated_text = r["generated_text"]
                for token_and_logprob in r["generated_tokens"]:
                    logprob = token_and_logprob.get("logprob", 0)
                    text = token_and_logprob["text"]
                    tokens.append(Token(text=text, logprob=logprob))
                    sequence_logprob += logprob

                completion = GeneratedOutput(text=generated_text, logprob=sequence_logprob, tokens=tokens)
                completions.append(completion)
        except Exception as e:
            hlog(f"GenerateInferenceHandler failed with exception {e} during parse_response {response}")
        return completions


class ChatModelInferenceHandler(ModelInferenceHandler[TextChatParameters]):
    def __init__(self, inference_engine: ModelInference):
        self.inference_engine = inference_engine

    def create_params(self, request: Request) -> TextChatParameters:
        return TextChatParameters(
            logprobs=True,
            presence_penalty=0,
            frequency_penalty=0,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

    def parse_response(self, response: dict) -> List[GeneratedOutput]:
        completions = []
        try:
            for raw_completion in response["choices"]:
                sequence_logprob: float = 0
                tokens: List[Token] = []
                generated_text = raw_completion["message"]["content"]

                for token_and_logprob in raw_completion["logprobs"]["content"]:
                    logprob = token_and_logprob["logprob"]
                    text = token_and_logprob["token"]
                    tokens.append(Token(text=text, logprob=logprob))
                    sequence_logprob += logprob

                completion = GeneratedOutput(text=generated_text, logprob=sequence_logprob, tokens=tokens)
                completions.append(completion)
        except Exception as e:
            hlog(f"ChatModelInferenceHandler failed with exception {e} during parse_response {response}")
        return completions

    def serve_request(self, prompt: str, params: TextChatParameters) -> Dict:
        semaphore = _get_semaphore(self.inference_engine.model_id)

        with semaphore:
            response = self.inference_engine.chat(
                messages=[{"role": "user", "content": prompt}],
                params=params,
            )
        return response


class IbmClient(CachingClient, ABC):
    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: str,
        region: str,
        location: dict,
        watsonx_model_name: str,
        **kwargs,
    ):
        super().__init__(cache_config=cache_config)
        self.project_id = None
        self.url = None
        self.watsonx_model_name = watsonx_model_name
        self.api_key = api_key
        self.region = region
        self.kwargs = kwargs
        for entry in location:
            if entry["region"].lower() == self.region.lower():
                self.project_id = entry["project_id"]
                self.url = entry["url"]

        assert self.project_id is not None, (
            "Missed project_id for specified region configuration in credentials.conf, should be in list "
            "of JSON objects with 'region', 'url', 'project_id' per region"
        )
        assert self.url is not None, (
            "Missed url for specified region configuration in credentials.conf, should be in list "
            "of JSON objects with 'region', 'url', 'project_id' per region"
        )

        self.inference_engine = ModelInference(
            model_id=self.watsonx_model_name,
            params={GenParams.MAX_NEW_TOKENS: 2000},
            credentials=Credentials(api_key=api_key, url=self.url),
            project_id=self.project_id,
        )

        hlog("Started IBM Client")

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        pass

    def do_call(self, inference_handler: ModelInferenceHandler, request: Request) -> RequestResult:
        params = inference_handler.create_params(request=request)

        def do_it() -> Dict[str, Any]:
            return inference_handler.serve_request(prompt=request.prompt, params=params)

        raw_request = {"prompt": request.prompt, "params": params.to_dict(), "model": request.model}

        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        completions = inference_handler.parse_response(response)
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )


class IbmChatClient(IbmClient):

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        try:
            return self.do_call(
                inference_handler=ChatModelInferenceHandler(inference_engine=self.inference_engine), request=request
            )

        except Exception as e:
            error: str = f"IBM Chat client Model error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])


class IbmTextClient(IbmClient):
    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        try:
            return self.do_call(
                inference_handler=GenerateInferenceHandler(inference_engine=self.inference_engine), request=request
            )
        except Exception as e:
            error: str = f"IBM Text client Model error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
