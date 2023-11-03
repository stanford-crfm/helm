import os
from typing import Any, Dict, Optional
from threading import Lock
from helm.common.cache import CacheConfig

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from helm.common.hierarchical_logger import htrack_block, hlog
from .caching_tokenizer import CachingTokenizer
from .tokenizer import cleanup_tokens


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


class HuggingFaceTokenizer(CachingTokenizer):
    _tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
    _tokenizers_lock: Lock = Lock()

    def __init__(
        self,
        cache_config: CacheConfig,
        pretrained_model_name_or_path: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._revision = revision

    @staticmethod
    def create_tokenizer(pretrained_model_name_or_path: str, revision: Optional[str] = None) -> PreTrainedTokenizerBase:
        """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
        # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
        # TODO: Figure out if we actually need this.
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        tokenizer_kwargs = {}
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        try:
            # From the Hugging Face documentation, "local_files_only(defaults to False) —
            # Whether or not to only look at local files".
            # Running `local_files_only=False` requires an internet connection even if the files are downloaded
            # and cached. We need to first run with `local_files_only=True` just in case the machine
            # we are running this code has connection issues. If the tokenizer files are not cached,
            # we attempt to download them from HuggingFace.
            # From https://huggingface.co/course/chapter6/3, "slow tokenizers are those written in Python inside
            # the Hugging Face Transformers library, while the fast versions are the ones provided by Hugging Face
            # Tokenizers, which are written in Rust." So, use the "fast" version of the tokenizers if available.
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, local_files_only=True, use_fast=True, **tokenizer_kwargs
            )
        except OSError:
            hlog(f"Local files do not exist for HuggingFace tokenizer: {pretrained_model_name_or_path}. Downloading...")
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, local_files_only=False, use_fast=True, **tokenizer_kwargs
            )

    @staticmethod
    def get_tokenizer(
        helm_tokenizer_name: str, pretrained_model_name_or_path: str, revision: Optional[str] = None
    ) -> PreTrainedTokenizerBase:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        with HuggingFaceTokenizer._tokenizers_lock:
            if helm_tokenizer_name not in HuggingFaceTokenizer._tokenizers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (revision={revision}) "
                    f"for HELM tokenizer {helm_tokenizer_name} with Hugging Face Transformers"
                ):
                    # Keep the tokenizer in memory, so we don't recreate it for future requests
                    HuggingFaceTokenizer._tokenizers[helm_tokenizer_name] = HuggingFaceTokenizer.create_tokenizer(
                        pretrained_model_name_or_path, revision
                    )
        return HuggingFaceTokenizer._tokenizers[helm_tokenizer_name]

    def _get_tokenizer_for_request(self, request: Dict[str, Any]) -> PreTrainedTokenizerBase:
        """Method used in both _tokenize_do_it and _decode_do_it to get the tokenizer."""
        pretrained_model_name_or_path: str
        if self._pretrained_model_name_or_path:
            pretrained_model_name_or_path = self._pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = resolve_alias(request["tokenizer"])
        _tokenizer = HuggingFaceTokenizer.get_tokenizer(
            helm_tokenizer_name=request["tokenizer"],
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=self._revision,
        )
        return _tokenizer

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        _tokenizer = self._get_tokenizer_for_request(request)

        if request["encode"]:
            if request["truncation"]:
                tokens = _tokenizer.encode(
                    request["text"],
                    truncation=request["truncation"],
                    max_length=request["max_length"],
                    add_special_tokens=False,
                )
            else:
                tokens = _tokenizer.encode(request["text"], add_special_tokens=False)
        else:
            if "gpt" in request["tokenizer"] or request["tokenizer"] in [
                "bigscience/bloom",
                "Writer/palmyra-base",
                "facebook/opt-66b",
            ]:
                # These models already handle the "▁" character correctly with the
                # convert_tokens_to_string method. We prefer to use this method instead
                # of the hacky cleanup_tokens method below as it might handle cases
                # we haven't thought of in cleanup_tokens.
                tokens = [
                    _tokenizer.convert_tokens_to_string([token]) for token in _tokenizer.tokenize(request["text"])
                ]
            else:
                # Tokenizes the text and returns the tokens as a list of strings,
                # not a list of token objects (otherwise "Hello world" would be"
                # ["Hello", "▁world"] and not ["Hello", " world"])
                # We could do this with a simple replace like this:
                # tokens = [_tokenizer.convert_tokens_to_string([i]) for i in _tokenizer.tokenize(request["text"])]
                # But this replaces all the "▁" characters by "", which is not what we want.
                # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                # Just like tokenize("Hello", encode=False) would return ["Hello"].
                tokens = _tokenizer.tokenize(request["text"])
                tokens = cleanup_tokens(tokens, request["tokenizer"])
        return {"tokens": tokens}

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        _tokenizer = self._get_tokenizer_for_request(request)

        text = _tokenizer.decode(
            request["tokens"], clean_up_tokenization_spaces=request["clean_up_tokenization_spaces"]
        )
        return {"text": text}
