from dataclasses import replace, asdict
from typing import Any, Dict, List, Optional, cast

from nemollm.api import NemoLLM
import os

import tiktoken
import requests
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token , EMBEDDING_UNAVAILABLE_REQUEST_RESULT

from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, truncate_sequence, wrap_request_time

access_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IlFVUTRNemhDUVVWQk1rTkJNemszUTBNMlFVVTRRekkyUmpWQ056VTJRelUxUTBVeE5EZzFNUSJ9.eyJpc3MiOiJodHRwczovL2F1dGgucGxhdGZvcm0uc3ltYmwuYWkvIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMDgxOTY5MjQ1NTg4NDYwMTMwMjAiLCJhdWQiOlsiaHR0cHM6Ly9hcGktbmVidWxhLnN5bWJsLmFpIiwiaHR0cHM6Ly9kaXJlY3QtcGxhdGZvcm0uYXV0aDAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5MDUzODcxNiwiZXhwIjoxNjkwNjI1MTE2LCJhenAiOiJWRmFjd2RQUzM3TFpBN2hqaER0cDhXVWxTTlhDY2RBYiIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwifQ.XGme0m3Oy9bz87ta_a_ZdtNLNTvG1ZDqUnTWpCrGMsDvem6cEbLz1noQk0dM6hX13t0vG627sIVeKZrZAQGlLSGOCL8MoOWRQJlFlKUBMlNlJhVg633hFz0_PpzudM2NuSNJlVul5q136_kQIQEgY4qTXKSaELOVCmW79YokuSusjjzIvwxaQ00sXfDdQKt2F-Na80B4nFFbeKYJgy9FwU3lf48L3IReW7bI-o2T-wcL8lAEvpIHr5U3N_yiONlkdYxppk3wRAZ4ZxJl5wHCisD--5PL0sGyi7p3b0peZtohYfP-PqdPxDNqlCiUPWoN1z8AASqh1iIBidNUSSY3YQ"

end_of_text_token = "<|endoftext|>"
result_token = "<|result|>"

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'  # Adjust the Content-Type based on your API's requirements
}

def create_token_chunks(transcript, tokenizer, max_token_per_chunk):
    if transcript is None:
        raise ValueError("Transcript cannot be None")

    transcript = transcript.strip()
    if len(transcript) <= 0:
        return []

    if '\n' in transcript:
        transcript_by_lines = transcript.split('\n')
    else:
        transcript_by_lines = [transcript]

    chunks = []
    last_chunk = []
    chunks.append(last_chunk)
    for index, line in enumerate(transcript_by_lines):
        line = line.strip()
        if len(line) <= 0:
            continue
        line_input_ids = tokenizer(line).input_ids
        line_input_ids.append(198)
        if len(line_input_ids) >= max_token_per_chunk:
            for i in range(0, len(line_input_ids), max_token_per_chunk):
                chunks.append(line_input_ids[i:i + max_token_per_chunk])
            continue

        if len(chunks) >= 1 and (len(last_chunk) + len(line_input_ids)) < max_token_per_chunk <= len(line_input_ids):
            last_chunk = chunks[-1]

        if len(last_chunk) + len(line_input_ids) >= max_token_per_chunk:
            last_chunk = []
            chunks.append(last_chunk)

        last_chunk.extend(line_input_ids)

    text_chunks = []
    for chunk in chunks:
        if len(chunk) <= 0:
            continue
        chunk_text = tokenizer.decode(chunk).replace('</s>', '\n').strip()
        text_chunks.append(chunk_text)

    return text_chunks


class FalconClient(Client):

    def __init__(
        self,
        cache_config: CacheConfig
    ):
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
          return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {
            "prompt" : f"{request.prompt}\n{end_of_text_token}\n{result_token}",
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "penalty_alpha" : request.penalty_alpha,
            "return_scores": False,
            "disable_logging" :True
        }
        print(raw_request)
        def do_it():
              API_HOST = "https://nebula.symbl.ai/v1-dev/generate"
              try: 
                  response = requests.request("POST", API_HOST, headers=headers, json=raw_request).json()
              except Exception as e:
                raise Exception(f"Error while processing request {raw_request} ---- {e}")
              if "output" not in response:
                  raise Exception("Invalid response from Falcon 40B API: " + str(response))
              return response

        try:
            cache_key = Client.make_cache_key({"engine": request.model_engine, **raw_request}, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            return RequestResult(success=False, cached=False, error=str(e), completions=[], embedding=[])

        raw_completion_content: str = response["output"]
        text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content

        tokenization_result: TokenizationRequestResult = self.tokenize(
            TokenizationRequest(
                text, tokenizer="tiiuae/falcon-40b"
            )
        )
        tokens = [Token(text=cast(str, raw_token), logprob=0, top_logprobs={})
                  for raw_token in tokenization_result.raw_tokens
                ]

        completions = [
            Sequence(
                text=response["output"],
                logprob=0,
                tokens=tokens,
            )
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    @staticmethod
    def _get_tokenizer_name(tokenizer: str) -> str:
        return tokenizer.split("/")[1]

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:

        cache_key = asdict(request)

        try:
            def do_it():
                tokens = tokenizer.encode(request.text)
                if not request.encode:
                    tokens = [tokenizer.decode([token]) for token in tokens]
                if request.truncation:
                    tokens = tokens[: request.max_length]
                return {"tokens": tokens}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

            result = TokenizationRequestResult(
                success=True,
                cached=cached,
                text=request.text,
                tokens=[TokenizationToken(value) for value in response["tokens"]],
                request_time=response["request_time"],
                error=None,
            )
            return result
        except Exception as error:
            raise ValueError(
                f"Error encoding with tiktoken: {error}."
            )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:

        cache_key = asdict(request)

        try:

            def do_it():
                tokens = [token if isinstance(token, int) else tokenizer.encode(token)[0] for token in request.tokens]
                text = tokenizer.decode(tokens)
                return {"text": text}

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

            return DecodeRequestResult(
                success=True,
                cached=cached,
                text=str(response["text"]),
                request_time=response["request_time"],
                error=None,
            )
        except Exception as error:
            raise ValueError(
                f"Error decoding with tiktoken: {error}."
            )

