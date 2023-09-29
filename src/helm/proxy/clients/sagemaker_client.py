from dataclasses import asdict
from typing import List, Optional, Dict

from filelock import FileLock
from openai.api_resources.abstract import engine_api_resource
import openai as turing

from helm.common.cache import Cache, CacheConfig
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult, TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from .huggingface_client import HuggingFaceServer
from .huggingface_tokenizer import HuggingFaceTokenizers
from .openai_client import ORIGINAL_COMPLETION_ATTRIBUTES

import boto3
import json

class SageMakerClient(Client):
    """
    Client for invoking a SageMaker Endpoint via direct boto3 invocation. Requires AWS Credentials
    to be provided via $aws configure or a similar method. SageMaker endpoint needs to be in the same SageMakerClient
    account and region as this client.
    """

    def __init__(self, sagemaker_endpoint_name: str, cache_config: CacheConfig,):
        self.sagemaker_endpoint_name = sagemaker_endpoint_name

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        print("in sagemaker make reqyest")
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT
        if self.sagemaker_endpoint_name is None:
            raise ValueError("SageMaker Endpoint name is required")

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
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

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        #model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        try:
            #payload example for gpt-2
            # payload example for gpt-j-6b
            # payload = {
            #     "text_inputs": raw_request["prompt"],
            #     "max_length": len(raw_request["prompt"]),
            #     "num_return_sequences": 1,
            #     "top_k": 3,
            #     "top_p": 0.95,
            #     "do_sample": True,
            #     "num_beams": 5,
            # }

            #runtime = boto3.Session().client('sagemaker-runtime')
            print("invoking endpoint")

            #payload example for Falcon
            payload = {
               "inputs": raw_request["prompt"],
               "parameters": {
                   "do_sample": True,
                   "top_p": 0.9,
                   "temperature": 0.8,
                   "max_new_tokens": 1024,
                   "return_full_text": False,
 #                  "stop": ["<|endoftext|>", "</s>"],
#                   "stop": ["\nQuestion","<|endoftext|>", "</s>"],
                   "stop": ["\n", "<|endoftext|>", "</s>"],

               },
            }

            encodedPayload = json.dumps(payload).encode("utf-8")
            print("Endpoint: ", self.sagemaker_endpoint_name)
            runtime = boto3.Session(region_name='us-west-2').client('sagemaker-runtime')
            response = runtime.invoke_endpoint(
                EndpointName=self.sagemaker_endpoint_name,
                ContentType='application/json',
                Body=encodedPayload
            )

            print(response)
            print("finish invoke")

            # Unpack response
            result = json.loads(response['Body'].read().decode())
            print("*****result is ")
            print(result)
            print("********")
            print("type", type(result))
            #def do_it():
            #    return model_server_instance.serve_request(raw_request)

            #cache_key = Client.make_cache_key(raw_request, request)
            #response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"SageMaker error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []

        #print("returning request result &&&&&&")
        #print(RequestResult)
        #print("&&&&&&")

        #completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)

        tokens = []
        # gpt models
        # for token_text in result[0][0]["generated_text"]:
        #     tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
        #completion = Sequence(text = result[0][0]["generated_text"], logprob=0.5, tokens=tokens)

        # Falcon
        for token_text in result[0]["generated_text"]:
            tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))

        completion = Sequence(text=result[0]["generated_text"], logprob=0.5, tokens=tokens)

        completions.append(completion)
        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            request_datetime=0,
            completions=completions,
            embedding=[],
        )



    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
#        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        tokenizer = HuggingFaceTokenizers.get_tokenizer("huggingface/gpt2")
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        # These models already handle the "▁" character correctly with the
                        # convert_tokens_to_string method. We prefer to use this method instead
                        # of the hacky cleanup_tokens method below as it might handle cases
                        # we haven't thought of in cleanup_tokens.
                        tokens = [
                            tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request.text)
                        ]
                    else:
                        # Tokenizes the text and returns the tokens as a list of strings,
                        # not a list of token objects (otherwise "Hello world" would be"
                        # ["Hello", "▁world"] and not ["Hello", " world"])
                        # We could do this with a simple replace like this:
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                        # But this replaces all the "▁" characters by "", which is not what we want.
                        # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                        # Just like tokenize("Hello", encode=False) would return ["Hello"].
                        tokens = tokenizer.tokenize(request.text)
                        tokens = cleanup_tokens(tokens, request.tokenizer)
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )


    #def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
    #    raise NotImplementedError("Use the HuggingFaceClient to tokenize.")


    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
