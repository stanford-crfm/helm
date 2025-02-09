from typing import Optional, Dict, Any, List
from threading import BoundedSemaphore
import json
from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
import urllib.request
import time

try:
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class AzureMLClient_old(OpenAIClient):
    """
    Client for accessing an Azure Machine Learning Inference endpoint with rate limiting.

    Configure by setting the following in `prod_env/credentials.conf`:

    ```
    azureMLEndpoint: https://azure.inference.ml.azure.com/score
    azureMLApiKey: your-private-key
    ```

    This client interacts with an Azure ML endpoint, ensuring requests are rate-limited to avoid 429 errors.
    """

    CREDENTIAL_HEADER_NAME = "Authorization"
    MAX_CONCURRENT_REQUESTS = 5  # Limit how many requests can be sent at a time
    RATE_LIMIT_DELAY = 0.5  # Delay in seconds between requests (adjust if needed)
    MAX_RETRIES = 20  # Number of retries for 429 errors
    semaphore = BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key="unused",  # Not needed for OpenAIClient
        )

        if not endpoint:
            raise NonRetriableException("Must provide Azure ML endpoint through credentials.conf")
        if not api_key:
            raise NonRetriableException("Must provide API key through credentials.conf")

        self.endpoint = endpoint.strip("/")
        self.api_key = api_key

    def _send_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to send a request to the Azure ML Inference endpoint with rate limiting and retries."""
        body = json.dumps({"input_data": input_data}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            self.CREDENTIAL_HEADER_NAME: f"Bearer {self.api_key}",
        }

        attempt = 0
        while attempt < self.MAX_RETRIES:
            with self.semaphore:  # Ensure we don't exceed concurrent request limits
                request = urllib.request.Request(self.endpoint, body, headers)

                try:
                    response = urllib.request.urlopen(request)
                    result = json.loads(response.read().decode("utf-8"))
                    return result

                except urllib.error.HTTPError as error:
                    if error.code == 429:  # Too Many Requests
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff (capped at 10s)
                        print(f"Rate limit exceeded (429). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                    else:
                        raise RuntimeError(
                            f"Request failed with status code: {error.code}\n"
                            f"Headers: {error.info()}\n"
                            f"Response: {error.read().decode('utf-8', 'ignore')}"
                        )

        raise RuntimeError("Max retries exceeded for 429 Too Many Requests")

    def make_request(self, request: Request) -> RequestResult:
        """
        Overrides the `make_request` function to send requests to the Azure ML endpoint.

        :param request: A `Request` object containing input data.
        :return: A `RequestResult` with the model's output.
        """
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        
        input_data = {
            "input_string": messages,
            "parameters": {
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "max_new_tokens": request.max_tokens or 512,
            },
        }

        response = self._send_request(input_data)

        # Extract model output
        model_response = response.get("output", "")

        if isinstance(model_response, list):  
            # Case 1: If Azure ML returns a list of responses, extract the first item's "content"
            text = model_response[0].get("content", "") if model_response else ""
        elif isinstance(model_response, str):  
            # Case 2: If Azure ML returns a raw string, use it directly
            text = model_response.strip()
        else:
            raise ValueError(f"Unexpected response format from Azure ML: {model_response}")

        # Tokenize the response
        tokenization_result = self.tokenizer.tokenize(
            TokenizationRequest(text, tokenizer=self.tokenizer_name)
        )
        tokens = [Token(text=t, logprob=0) for t in tokenization_result.raw_tokens]

        completion = GeneratedOutput(
            text=text,
            logprob=0,  # Azure ML may not provide log probabilities
            tokens=tokens,
            finish_reason={"reason": "stop"},
        )

        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            completions=[completion],
            embedding=[],
        )


from threading import BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

class AzureMLClient(OpenAIClient):
    """
    Client for accessing an Azure Machine Learning Inference endpoint with rate limiting.

    Configure by setting the following in `prod_env/credentials.conf`:

    ```
    azureMLEndpoint: https://azure.inference.ml.azure.com/score
    azureMLApiKey: your-private-key
    ```

    This client interacts with an Azure ML endpoint, ensuring requests are rate-limited to avoid 429 errors.
    """

    CREDENTIAL_HEADER_NAME = "Authorization"
    MAX_CONCURRENT_REQUESTS = 500  # Limit how many requests can be sent at a time
    RATE_LIMIT_DELAY = 0.5  # Delay in seconds between requests (adjust if needed)
    MAX_RETRIES = 25  # Number of retries for 429 errors
    semaphore = BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key="unused",  # Not needed for OpenAIClient
        )

        if not endpoint:
            raise NonRetriableException("Must provide Azure ML endpoint through credentials.conf")
        if not api_key:
            raise NonRetriableException("Must provide API key through credentials.conf")

        self.endpoint = endpoint.strip("/")
        self.api_key = api_key

    def _send_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to send a request to the Azure ML Inference endpoint with rate limiting and retries."""
        body = json.dumps({"input_data": input_data}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            self.CREDENTIAL_HEADER_NAME: f"Bearer {self.api_key}",
        }

        attempt = 0
        while attempt < self.MAX_RETRIES:
            with self.semaphore:  # Ensure we don't exceed concurrent request limits
                request = urllib.request.Request(self.endpoint, body, headers)

                try:
                    response = urllib.request.urlopen(request)
                    result = json.loads(response.read().decode("utf-8"))
                    return result

                except urllib.error.HTTPError as error:
                    if error.code == 429:  # Too Many Requests
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff (capped at 10s)
                        print(f"Rate limit exceeded (429). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                    else:
                        raise RuntimeError(
                            f"Request failed with status code: {error.code}\n"
                            f"Headers: {error.info()}\n"
                            f"Response: {error.read().decode('utf-8', 'ignore')}"
                        )

        raise RuntimeError("Max retries exceeded for 429 Too Many Requests")

    def make_request(self, request: Request) -> RequestResult:
        """
        Overrides the `make_request` function to send requests to the Azure ML endpoint.

        :param request: A `Request` object containing input data.
        :return: A `RequestResult` with the model's output.
        """
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        
        input_data = {
            "input_string": messages,
            "parameters": {
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "max_new_tokens": request.max_tokens or 512,
            },
        }

        response = self._send_request(input_data)

        # Extract model output
        model_response = response.get("output", "")

        if isinstance(model_response, list):  
            # Case 1: If Azure ML returns a list of responses, extract the first item's "content"
            text = model_response[0].get("content", "") if model_response else ""
        elif isinstance(model_response, str):  
            # Case 2: If Azure ML returns a raw string, use it directly
            text = model_response.strip()
        else:
            raise ValueError(f"Unexpected response format from Azure ML: {model_response}")

        # Tokenize the response
        tokenization_result = self.tokenizer.tokenize(
            TokenizationRequest(text, tokenizer=self.tokenizer_name)
        )
        tokens = [Token(text=t, logprob=0) for t in tokenization_result.raw_tokens]

        completion = GeneratedOutput(
            text=text,
            logprob=0,  # Azure ML may not provide log probabilities
            tokens=tokens,
            finish_reason={"reason": "stop"},
        )

        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            completions=[completion],
            embedding=[],
        )

    def process_batch_requests(self, requests: List[Request]) -> List[RequestResult]:
        """
        Processes multiple requests while respecting concurrency limits.

        :param requests: List of `Request` objects to process.
        :return: List of `RequestResult` responses.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as executor:
            futures = {executor.submit(self.make_request, req): req for req in requests}

            for future in as_completed(futures):
                result = future.result()  # Retrieve response from each request
                results.append(result)

        return results
