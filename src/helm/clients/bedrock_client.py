from abc import abstractmethod
from copy import deepcopy
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from helm.common.cache import CacheConfig
from helm.clients.client import CachingClient, truncate_and_tokenize_response_text
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.clients.bedrock_utils import get_bedrock_client
from helm.tokenizers.tokenizer import Tokenizer


JSON_CONTENT_TYPE = "application/json"


class BedrockClient(CachingClient):
    @abstractmethod
    def convert_request_to_raw_request(self, request: Request) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def convert_raw_response_to_completions(self, response: Dict, request: Request) -> List[GeneratedOutput]:
        raise NotImplementedError()

    def __init__(
        self,
        cache_config: CacheConfig,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        bedrock_model_id: Optional[str] = None,
        assumed_role: Optional[str] = None,
        region: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.bedrock_model_id = bedrock_model_id
        self.bedrock_client = get_bedrock_client(
            assumed_role=assumed_role or os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=region or os.environ.get("AWS_DEFAULT_REGION", None),
        )

    def make_request(self, request: Request) -> RequestResult:
        # model_id should be something like "amazon.titan-tg1-large"
        model_id = self.bedrock_model_id if self.bedrock_model_id else request.model.replace("/", ".")
        raw_request = self.convert_request_to_raw_request(request)

        # modelId isn't part of raw_request, so it must be explicitly passed into the input to
        raw_request_for_cache: Dict = {"modelId": model_id, **deepcopy(raw_request)}
        cache_key: Mapping = CachingClient.make_cache_key(raw_request_for_cache, request)

        def do_it() -> Dict[Any, Any]:
            response = self.bedrock_client.invoke_model(
                body=json.dumps(raw_request), modelId=model_id, accept=JSON_CONTENT_TYPE, contentType=JSON_CONTENT_TYPE
            )
            return json.loads(response.get("body").read())

        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as error:
            return RequestResult(
                success=False,
                cached=False,
                error=str(error),
                completions=[],
                embedding=[],
            )

        completions = self.convert_raw_response_to_completions(response, request)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response["request_datetime"],
            completions=completions,
            embedding=[],
        )


class BedrockTitanClient(BedrockClient):
    _COMPLETION_REASON_TO_FINISH_REASON = {
        "LENGTH": "length",
        "FINISH": "endoftext",
    }

    def convert_request_to_raw_request(self, request: Request) -> Dict:
        # TODO: Support the following:
        # - top_k_per_token
        # - echo_prompt
        # - num_completions
        return {
            "inputText": request.prompt,
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens,
                # We ignore stop sequences in the request and always set stop sequences to the empty list.
                # This is because:
                #
                # 1. The only permitted stop sequences are "|" and "User:"
                #     - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
                #     - https://github.com/boto/boto3/issues/3993
                #     - https://github.com/aws/aws-sdk/issues/692
                #
                # 2. Titan has the tendency to emit "\n" as the first token in the generated text output,
                #    which would cause the output to stop immediately if "\n" is in the stop_sequences.
                "stopSequences": [],
                "temperature": request.temperature,
                "topP": request.top_p,
            },
        }

    def convert_raw_response_to_completions(self, response: Dict, request: Request) -> List[GeneratedOutput]:
        # TODO: Support the following:
        # - tokens
        # - logprob
        completions: List[GeneratedOutput] = []
        for raw_completion in response["results"]:
            output_text = raw_completion["outputText"]
            # Call lstrip() Titan has the tendency to emit "\n" as the first token in the generated text output.
            finish_reason = BedrockTitanClient._COMPLETION_REASON_TO_FINISH_REASON.get(
                raw_completion["completionReason"], raw_completion["completionReason"].lower()
            )
            completion = truncate_and_tokenize_response_text(
                output_text.lstrip(), request, self.tokenizer, self.tokenizer_name, finish_reason
            )
            completions.append(completion)
        return completions
