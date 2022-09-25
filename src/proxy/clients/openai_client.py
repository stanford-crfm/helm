from typing import List, Optional, Dict

import openai
import requests
from dacite import from_dict
from googleapiclient.errors import BatchError, HttpError
from httplib2 import HttpLib2Error

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time
from common.openai_moderation_request import ModerationAttributes, OpenAIModerationAPIRequestResult

OPENAI_END_OF_TEXT_TOKEN: str = "<|endoftext|>"
ORIGINAL_COMPLETION_ATTRIBUTES = openai.api_resources.completion.Completion.__bases__


class OpenAIClient(Client):
    def __init__(self, api_key: str, cache_path: str, org_id: Optional[str] = None):
        self.org_id: Optional[str] = org_id
        self.api_key: str = api_key
        self.api_base: str = "https://api.openai.com"
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "echo": request.echo_prompt,
        }

        # OpenAI doesn't let you ask for more completions than the number of
        # per-token candidates.
        raw_request["best_of"] = max(raw_request["best_of"], raw_request["n"])
        raw_request["logprobs"] = max(raw_request["logprobs"], raw_request["n"])

        try:

            def do_it():
                # Following https://beta.openai.com/docs/api-reference/authentication
                # `organization` can be set to None.
                openai.organization = self.org_id
                openai.api_key = self.api_key
                openai.api_base = self.api_base
                openai.api_resources.completion.Completion.__bases__ = ORIGINAL_COMPLETION_ATTRIBUTES
                return openai.Completion.create(**raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.error.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[])

        completions = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for text, logprob, top_logprobs in zip(
                raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
            ):
                # Do not include these excess tokens in the response.
                # TODO: this is a hacky solution until we figure out why
                #       OpenAI is sending tokens including and past the stop sequences.
                # TODO: This logic doesn't work when the stop sequences spans multiple tokens.
                #       https://github.com/stanford-crfm/benchmarking/issues/53
                if any(stop in text for stop in request.stop_sequences):
                    break

                tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                sequence_logprob += logprob or 0
            completion = Sequence(
                text=raw_completion["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")

    def extract_moderation_scores(response: Dict) -> ModerationAttributes:
        categories = response["results"][0]["categories"].values()
        category_scores = response["results"][0]["category_scores"].values()
        category_names = [key.translate(key.maketrans("/-", "__")) + "_score" for key in response["results"][0]["category_scores"].keys()]
        values = [{"is_moderated": x, "score": y} for x, y in list(zip(categories, category_scores))]
        
        all_scores = dict(zip(category_names, values))
        return from_dict(data_class=ModerationAttributes, data=all_scores)

    # Inputting a single example, but returning a dict -> handle this in moderation_metrics
    def get_moderation_scores(self, input) -> OpenAIModerationAPIRequestResult: 
        """
        Make call to OpenAI Moderation API. 
        """
        request = {"input": input}
        try:
            def do_it():
                text_to_response: Dict[str, Dict] = dict()

                auth_string = "Bearer "+self.api_key
                headers = {'content-type': 'application/json', "Authorization": auth_string}
                response = requests.post(url = "https://api.openai.com/v1/moderations", json=request, headers=headers)
                
                text_to_response[input] = response

                return text_to_response

        except (BatchError, HttpLib2Error, HttpError) as e:
            return OpenAIModerationAPIRequestResult(
                success=False, cached=False, error=f"Error was thrown when making a request to OpenAI Moderation API: {e}"
            )
            # raise Exception(f"Error was thrown when making a request to OpenAI Moderation API: ", e)
        
        response, cached = self.cache.get(request, do_it)
        response_json = response[input].json()
        moderation_attributes = OpenAIClient.extract_moderation_scores(response_json)
        return OpenAIModerationAPIRequestResult(
            success=True,
            cached=cached,
            flagged=response_json["results"][0]["flagged"],
            moderation_attributes=moderation_attributes
        )