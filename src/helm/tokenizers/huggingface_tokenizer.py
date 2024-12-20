import os
from typing import Any, Dict, Optional, cast
from threading import Lock
from helm.common.cache import CacheConfig
from helm.common.concurrency import ThreadSafeWrapper

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from helm.common.hierarchical_logger import htrack_block, hlog
from helm.tokenizers.caching_tokenizer import CachingTokenizer
from helm.tokenizers.tokenizer import cleanup_tokens


WrappedPreTrainedTokenizer = ThreadSafeWrapper[PreTrainedTokenizerBase]
"""Thread safe wrapper around Hugging Face PreTrainedTokenizerBase.

Hugging Face PreTrainedTokenizerBase is thread-hostile and using it from multiple threads
simultaneously can result in an "Already borrowed" error (#1421). This wrapper ensures
that a lock is held when using the PreTrainedTokenizerBase.

Example usage:

    with wrapped_tokenizer as tokenizer:
        tokenizer.encode("...")
"""


class HuggingFaceTokenizer(CachingTokenizer):
    _tokenizers: Dict[str, WrappedPreTrainedTokenizer] = {}
    _tokenizers_lock: Lock = Lock()

    def __init__(
        self,
        cache_config: CacheConfig,
        tokenizer_name: str,
        pretrained_model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(cache_config=cache_config)
        self._helm_tokenizer_name = (
            tokenizer_name  # HELM tokenizer name (e.g. "huggingface/gpt2"), *not* Hugging Face Hub Model ID
        )
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._kwargs = kwargs

    @staticmethod
    def create_tokenizer(pretrained_model_name_or_path: str, **kwargs) -> WrappedPreTrainedTokenizer:
        """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
        # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
        # TODO: Figure out if we actually need this.
        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        from_pretrained_kwargs = {**kwargs}
        # If unspecified, set `use_fast=True` by default.
        if "use_fast" not in from_pretrained_kwargs:
            from_pretrained_kwargs["use_fast"] = True
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
            return WrappedPreTrainedTokenizer(
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path, local_files_only=True, **from_pretrained_kwargs
                )
            )
        except OSError:
            hlog(f"Local files do not exist for HuggingFace tokenizer: {pretrained_model_name_or_path}. Downloading...")
            return WrappedPreTrainedTokenizer(
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path, local_files_only=False, **from_pretrained_kwargs
                )
            )

    @staticmethod
    def get_tokenizer(
        helm_tokenizer_name: str, pretrained_model_name_or_path: str, **kwargs
    ) -> WrappedPreTrainedTokenizer:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        with HuggingFaceTokenizer._tokenizers_lock:
            if helm_tokenizer_name not in HuggingFaceTokenizer._tokenizers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (kwargs={kwargs}) "
                    f"for HELM tokenizer {helm_tokenizer_name} with Hugging Face Transformers"
                ):
                    # Keep the tokenizer in memory, so we don't recreate it for future requests
                    HuggingFaceTokenizer._tokenizers[helm_tokenizer_name] = HuggingFaceTokenizer.create_tokenizer(
                        pretrained_model_name_or_path, **kwargs
                    )
        return HuggingFaceTokenizer._tokenizers[helm_tokenizer_name]

    def get_wrapped_tokenizer(self) -> WrappedPreTrainedTokenizer:
        """Get the underlying Hugging Face WrappedPreTrainedTokenizer."""
        pretrained_model_name_or_path = (
            self._pretrained_model_name_or_path if self._pretrained_model_name_or_path else self._helm_tokenizer_name
        )
        return HuggingFaceTokenizer.get_tokenizer(
            helm_tokenizer_name=self._helm_tokenizer_name,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **self._kwargs,
        )

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request["tokenizer"] != self._helm_tokenizer_name:
            raise ValueError(
                f"This HuggingFaceTokenizer expects tokenizer to be {self._helm_tokenizer_name} "
                "but instead the request has tokenizer {request['tokenizer']}"
            )
        if request["encode"]:
            if request["truncation"]:
                with self.get_wrapped_tokenizer() as tokenizer:
                    tokens = tokenizer.encode(
                        request["text"],
                        truncation=request["truncation"],
                        max_length=request["max_length"],
                        add_special_tokens=False,
                    )
            else:
                with self.get_wrapped_tokenizer() as tokenizer:
                    tokens = tokenizer.encode(request["text"], add_special_tokens=False)
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
                with self.get_wrapped_tokenizer() as tokenizer:
                    tokens = [
                        tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request["text"])
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
                with self.get_wrapped_tokenizer() as tokenizer:
                    tokens = tokenizer.tokenize(request["text"])
                # Some tokenizers (e.g. Qwen/Qwen-7B) return the tokens as bytes, so we have to decode them to strings.
                if tokens and type(tokens[0]) == bytes:
                    tokens = [cast(bytes, token).decode(errors="ignore") for token in tokens]
                tokens = cleanup_tokens(tokens, request["tokenizer"])
        return {"tokens": tokens}

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request["tokenizer"] != self._helm_tokenizer_name:
            raise ValueError(
                f"This HuggingFaceTokenizer expects tokenizer to be {self._helm_tokenizer_name} "
                "but instead the request has tokenizer {request['tokenizer']}"
            )
        with self.get_wrapped_tokenizer() as tokenizer:
            text = tokenizer.decode(
                request["tokens"], clean_up_tokenization_spaces=request["clean_up_tokenization_spaces"]
            )
        return {"text": text}
