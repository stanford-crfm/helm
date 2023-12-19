from abc import abstractmethod
from copy import deepcopy
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from helm.common.cache import CacheConfig
from helm.proxy.clients.client import CachingClient
from helm.common.request import Request, RequestResult, Sequence, Token, wrap_request_time
from helm.proxy.clients.bedrock_utils import get_bedrock_client


JSON_CONTENT_TYPE = "application/json"


class BedrockClient(CachingClient):
    @abstractmethod
    def convert_request_to_raw_request(self, request: Request) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def convert_raw_response_to_completions(self, response: Dict) -> List[Sequence]:
        raise NotImplementedError()

    def __init__(
        self,
        cache_config: CacheConfig,
        bedrock_model_id: Optional[str] = None,
        assumed_role: Optional[str] = None,
        region: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
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

        completions = self.convert_raw_response_to_completions(response)

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

    def convert_raw_response_to_completions(self, response: Dict) -> List[Sequence]:
        # TODO: Support the following:
        # - tokens
        # - logprob
        completions: List[Sequence] = []
        for raw_completion in response["results"]:
            output_text = raw_completion["outputText"]
            tokens = [Token(text=output_text, logprob=0.0, top_logprobs={})]
            completion = Sequence(
                text=output_text,
                logprob=0.0,
                tokens=tokens,
                finish_reason={
                    "reason": BedrockTitanClient._COMPLETION_REASON_TO_FINISH_REASON.get(
                        raw_completion["completionReason"], raw_completion["completionReason"].lower()
                    )
                },
            )
            completions.append(completion)
        return completions
