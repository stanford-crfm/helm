import uuid
from abc import ABC
from abc import abstractmethod

# from collections import defaultdict

from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    TextGenParameters,
    # TextGenDecodingMethod,
    # TextGenLengthPenalty,
    ReturnOptionProperties,
)

from helm.common.hierarchical_logger import htrack_block, hlog
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
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from typing import Any, Dict, List, TypedDict, Callable, Union
from threading import Lock, Semaphore
import threading

# import time

# Define the maximum number of parallel executions is limited by IBM API
MAX_CONCURRENT_REQUESTS = 4
__semaphores: Dict[str, Semaphore] = dict()


def get_semaphore(model: str):
    if model not in __semaphores:
        __semaphores[model] = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

    return __semaphores[model]

class IBMRequest(TypedDict):
    """Data passed between make_request and serve_request. Used as the cache key."""
    model :str
    engine: str
    prompt: str
    temperature: float
    num_return_sequences: int
    max_new_tokens: int
    top_p: float
    echo_prompt: bool
    top_k_per_token: int
    stop_sequences: List


class ModelInferenceHandler(ABC):
    @abstractmethod
    def __init__(self, inference_engine: ModelInference):
        """
        :type inference_engine: object
        """
        self.inference_engine = inference_engine



    @abstractmethod
    def serve_request(self, raw_request: IBMRequest) -> Dict:
        pass

    @abstractmethod
    def parse_response(self, response: dict) -> List[GeneratedOutput]:
        pass

    @abstractmethod
    def create_params(self, request: IBMRequest) -> Union[TextGenParameters, TextChatParameters]:
        pass

    # @abstractmethod
    # def tokenize(self, text: str) -> dict:
    #     pass

    @staticmethod
    def pre_processing(text: str) -> str:
        """

        :rtype: object
        """
        return text.encode("unicode_escape").decode("utf-8")


class GenerateInferenceHandler(ModelInferenceHandler):

    def __init__(self, inference_engine: ModelInference):
        self.inference_engine = inference_engine





    def create_params(self, raw_request: IBMRequest) -> TextGenParameters:
        return TextGenParameters(
            # decoding_method=TextGenDecodingMethod.GREEDY,
            # length_penalty=TextGenLengthPenalty.get_sample_params(),
            temperature=0.05,
            top_p=raw_request["top_p"],
            # top_k = 5,
            # random_seed = 42,
            # repetition_penalty = 1.7,
            # min_new_tokens = 3,
            max_new_tokens=raw_request["max_new_tokens"],
            # stop_sequences = None,
            # time_limit = None,
            # truncate_input_tokens = 300,
            return_options=ReturnOptionProperties(
                input_text=True,
                generated_tokens=True,
                input_tokens=False,
                token_logprobs=True,
                token_ranks=False,
                # top_n_tokens = 1
            ),
            include_stop_sequence=False,
            prompt_variables=None,
        )

    def serve_request(self, raw_request: IBMRequest) -> Dict:
        response = self.inference_engine.generate(
            prompt=GenerateInferenceHandler.pre_processing(raw_request["prompt"]),
            params=self.create_params(raw_request),
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

    def tokenize(self, text: str) -> dict:
        tokens = {}
        try:
            tokens = self.inference_engine.tokenize(GenerateInferenceHandler.pre_processing(text), return_tokens=True)
        except Exception as e:
            hlog(f"GenerateInferenceHandler : Tokenization failed with exception {e} during tokenization of  {text}")

        return tokens


class ChatModelInferenceHandler(ModelInferenceHandler):
    # _lock = threading.Lock()  # Lock to protect the counter
    def __init__(self, inference_engine: ModelInference):
        self.inference_engine = inference_engine

    def active_threads(self):
        active_semaphore_threads = IBMServer._semaphore._value
        return MAX_CONCURRENT_REQUESTS - active_semaphore_threads

    def create_params(self, raw_request: IBMRequest) -> TextChatParameters:
        return TextChatParameters(
            logprobs=True,
            presence_penalty=0,
            frequency_penalty=0,
            temperature=raw_request["temperature"],
            max_tokens=raw_request["max_new_tokens"],
            top_p=raw_request["top_p"],
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

    def serve_request(self, raw_request: IBMRequest) -> Dict:
        semaphore = get_semaphore(raw_request["model"])

        with semaphore:
            tid = str(uuid.uuid4().hex)[:8]
            hlog(f"ENTER Active threads : {MAX_CONCURRENT_REQUESTS - semaphore._value}")
            hlog(f"Request: {tid} - {raw_request['prompt']}")
            response = self.inference_engine.chat(
                messages=[{"role": "user", "content": GenerateInferenceHandler.pre_processing(raw_request["prompt"])}],
                params=self.create_params(raw_request),
            )
            hlog(f"Response :{tid} - {response}")
        hlog(f"EXIT Active threads : {MAX_CONCURRENT_REQUESTS - semaphore._value}")
        return response

    # def tokenize(self, text: str) -> dict:
    #     semaphore = get_semaphore(raw_request["model"])
    #     with semaphore:
    #         try:
    #             return self.inference_engine.tokenize(GenerateInferenceHandler.pre_processing(text), return_tokens=True)
    #         except Exception as e:
    #             hlog(f"ChatModelInferenceHandler : Tokenization failed with exception {e} during tokenization of {text}")
    #     return {}


class IBMServer:
    _semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
    # _lock = threading.Lock()  # Lock to protect the counter
    # _method_counts: DefaultDict[str, int] = defaultdict(int)

    def __init__(
        self, model_name: str, api_key: str, project_id: str, location: str, inference_handler: Callable, **kwargs
    ):
        self.model_name = model_name

        with htrack_block(f"Loading IBM model {model_name} {location}"):
            model = ModelInference(
                model_id=model_name,
                params={GenParams.MAX_NEW_TOKENS: 2000},
                credentials=Credentials(api_key=api_key, url=location),
                project_id=project_id,
            )
            self.__inference_handler: ModelInferenceHandler = inference_handler(inference_engine=model)

    # def counter_key(self, method) -> str:
    #     return f"{self.model_name}:{method}"

    # def log_method_call(self, method_name: str):
    #     IBMServer._method_counts[self.counter_key(method_name)] += 1

    # def get_log_method_counter(self, method_name: str) -> int:
    #     return IBMServer._method_counts[self.counter_key(method_name)]

    # def active_threads(self):
    #     active_semaphore_threads = IBMServer._semaphore._value
    #     return MAX_CONCURRENT_REQUESTS - active_semaphore_threads

    # def _log_active_threads(self, thread_id: int, method_name: str, duration=None, data: str = None):
    #     if duration:  # state > EXIT
    #         hlog(
    #             f"{self.get_log_method_counter(method_name)} [{method_name}:{self.model_name}]
    #             {len(data)} < Completed in {duration:.4f} seconds >"
    #         )

    # def _enter_api(self, thread_id: int, method_name: str):
    #     with IBMServer._lock:
    #         self.log_method_call(method_name=method_name)
    #         hlog(f"Enter : Active threads : {self.active_threads()}")

    # def _exit_api(self, thread_id: int, method_name: str, start_time, data: str):
    #     duration = time.perf_counter() - start_time
    #     with IBMServer._lock:
    #         self._log_active_threads(thread_id=thread_id, method_name=method_name, duration=duration, data=data)
    #         hlog(f"Exit : Active threads : {self.active_threads()}")

    # def encode(self, text: str, **kwargs) -> List[int]:
    #     # start_time = time.perf_counter()
    #     with IBMServer._semaphore:
    #         # tid = threading.get_ident()
    #         # self._enter_api(thread_id=tid, method_name="encode")
    #         encoding_result = self.__inference_handler.tokenize(text)
    #         # self._exit_api(thread_id=tid, method_name="encode", start_time=start_time, data=text)
    #         return encoding_result["result"]["tokens"]

    def parse_response(self, response: dict) -> List[GeneratedOutput]:
        return self.__inference_handler.parse_response(response)

    def serve_request(self, raw_request: IBMRequest) -> Dict:
        # start_time = time.perf_counter()
        with IBMServer._semaphore:
            # tid = threading.get_ident()
            # self._enter_api(tid, "serve_request")
            response = self.__inference_handler.serve_request(raw_request=raw_request)
            # self._exit_api(tid, "serve_request", start_time=start_time, data=raw_request["prompt"])
            return response


class IBMServerFactory:
    """A factory that creates and caches IBMServer objects."""

    _servers: Dict[str, IBMServer] = {}
    _servers_lock: Lock = Lock()

    @staticmethod
    def get_server(
        model_name: str,
        ibm_model_name: str,
        api_key: str,
        project_id: str,
        location: str,
        inference_handler: Callable,
        **kwargs,
    ) -> Any:
        """
        Checks if the desired Model is cached. Creates the Model if it's not cached.
        Returns the Model.
        """
        with IBMServerFactory._servers_lock:
            if model_name not in IBMServerFactory._servers:
                with htrack_block(f"Creation model {model_name} and saving in cache with params  (kwargs={kwargs}) "):
                    IBMServerFactory._servers[model_name] = IBMServer(
                        model_name=ibm_model_name,
                        api_key=api_key,
                        project_id=project_id,
                        location=location,
                        inference_handler=inference_handler,
                        **kwargs,
                    )

        return IBMServerFactory._servers[model_name]


class IbmClient(CachingClient, ABC):
    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: str,
        region: str,
        location: str,
        inner_model_name: str,
        **kwargs,
    ):
        super().__init__(cache_config=cache_config)
        self.project_id = None
        self.url = None
        self.inner_model_name = inner_model_name
        self.api_key = api_key
        self.region = region
        self.kwargs = kwargs
        self.load_project_auth(location)

    def load_project_auth(self, location):
        for entry in location:
            if entry["region"].lower() == self.region.lower():
                self.project_id = entry["project_id"]
                self.url = entry["url"]

    def get_model_inference(self, model_name: str, api_key: str, location: str, project_id: str) -> ModelInference:
        with htrack_block(f"Loading IBM model {model_name} {location}"):
            return  ModelInference(
                                    model_id=model_name,
                                    params={GenParams.MAX_NEW_TOKENS: 2000},
                                    credentials=Credentials(api_key=api_key, url=location),
                                    project_id=project_id,
                                  )

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        pass


class IbmChatClient(IbmClient):

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request: IBMRequest = {
            "model" : self.inner_model_name,
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        try:
            inference_engine = self.get_model_inference(
                                                            model_name=self.inner_model_name,
                                                            api_key=self.api_key,
                                                            project_id=self.project_id,
                                                            location=self.url,
                                                        )
            chat_inference_handler = ChatModelInferenceHandler(inference_engine=inference_engine)
            # ibm_model: IBMServer = IBMServerFactory.get_server(
            #     model_name=request.model,
            #     ibm_model_name=self.inner_model_name,
            #     api_key=self.api_key,
            #     project_id=self.project_id,
            #     location=self.url,
            #     inference_handler=ChatModelInferenceHandler,
            #     **self.kwargs,
            # )

            def do_it() -> Dict[str, Any]:
                return chat_inference_handler.serve_request(raw_request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            completions = chat_inference_handler.parse_response(response)
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response["request_time"],
                request_datetime=response.get("request_datetime"),
                completions=completions,
                embedding=[],
            )

        except Exception as e:
            error: str = f"IBM Chat client Model error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])


class IbmTextClient(IbmClient):
    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request: IBMRequest = {
            "model": self.inner_model_name,
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        try:
            ibm_model: IBMServer = IBMServerFactory.get_server(
                model_name=request.model,
                ibm_model_name=self.inner_model_name,
                api_key=self.api_key,
                project_id=self.project_id,
                location=self.url,
                inference_handler=GenerateInferenceHandler,
                **self.kwargs,
            )

            def do_it() -> Dict[str, Any]:
                return ibm_model.serve_request(raw_request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            completions = ibm_model.parse_response(response)
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response["request_time"],
                request_datetime=response.get("request_datetime"),
                completions=completions,
                embedding=[],
            )

        except Exception as e:
            error: str = f"IBM Text client Model error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
