from copy import deepcopy
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Optional

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from .huggingface_tokenizer import HuggingFaceTokenizers
from threading import Lock


# TODO: Delete this.
_MODEL_NAME_ALIASES: Dict[str, str] = {
    "google/t5-11b": "t5-11b",
    "huggingface/gpt2": "gpt2",
    "huggingface/santacoder": "bigcode/santacoder",
    "huggingface/starcoder": "bigcode/starcoder",
}
"""Mapping of some HELM model names to Hugging Face pretrained model name."""


# TODO: Delete this.
def resolve_alias(model_name: str) -> str:
    """Resolve some HELM model names to Hugging Face pretrained model name."""
    return _MODEL_NAME_ALIASES.get(model_name, model_name)


class HuggingFaceServer:
    """A thin wrapper around a Hugging Face AutoModelForCausalLM for HuggingFaceClient to call."""

    def __init__(self, pretrained_model_name_or_path: str, revision: Optional[str] = None):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        model_kwargs = {}
        if revision:
            model_kwargs["revision"] = revision
        with htrack_block(f"Loading Hugging Face model {pretrained_model_name_or_path}"):
            # WARNING this may fail if your GPU does not have enough memory
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True, **model_kwargs
            ).to(self.device)
        with htrack_block(f"Loading Hugging Face tokenizer for model {pretrained_model_name_or_path}"):
            self.tokenizer: AutoTokenizer = HuggingFaceTokenizers.create_tokenizer(
                pretrained_model_name_or_path, revision
            )

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
            self.device
        )
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(
                raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
            )
            assert len(stop_sequence_ids.input_ids) == 1, "Total number of stop words should be 1."
            assert len(stop_sequence_ids.input_ids[0]) == 1, "Total number of tokens in each stop word should be 1."
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # Use HuggingFace's `generate` method.
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


class HuggingFaceServerFactory:
    """A factory that creates and caches HuggingFaceServer objects."""

    _servers: Dict[str, HuggingFaceServer] = {}
    _servers_lock: Lock = Lock()

    @staticmethod
    def get_server(helm_model_name: str, pretrained_model_name_or_path: str, revision: Optional[str] = None) -> Any:
        """
        Checks if the desired HuggingFaceModel is cached. Creates the HuggingFaceModel if it's not cached.
        Returns the HuggingFaceModel.
        """
        with HuggingFaceServerFactory._servers_lock:
            if helm_model_name not in HuggingFaceServerFactory._servers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (revision={revision}) "
                    f"for HELM model {helm_model_name} with Hugging Face Transformers"
                ):
                    HuggingFaceServerFactory._servers[helm_model_name] = HuggingFaceServer(
                        pretrained_model_name_or_path, revision
                    )

        return HuggingFaceServerFactory._servers[helm_model_name]


class HuggingFaceClient(Client):
    def __init__(
        self,
        cache_config: CacheConfig,
        pretrained_model_name_or_path: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.cache = Cache(cache_config)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._revision = revision

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

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

        pretrained_model_name_or_path: str
        if self._pretrained_model_name_or_path:
            pretrained_model_name_or_path = self._pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = resolve_alias(request.model)
        huggingface_model: HuggingFaceServer = HuggingFaceServerFactory.get_server(
            helm_model_name=request.model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=self._revision,
        )

        try:

            def do_it():
                return huggingface_model.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        pretrained_model_name_or_path: str
        if self._pretrained_model_name_or_path:
            pretrained_model_name_or_path = self._pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = resolve_alias(request.tokenizer)
        tokenizer = HuggingFaceTokenizers.get_tokenizer(
            helm_tokenizer_name=request.tokenizer,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=self._revision,
        )
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

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        pretrained_model_name_or_path: str
        if self._pretrained_model_name_or_path:
            pretrained_model_name_or_path = self._pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = resolve_alias(request.tokenizer)
        tokenizer = HuggingFaceTokenizers.get_tokenizer(
            helm_tokenizer_name=request.tokenizer,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=self._revision,
        )
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
