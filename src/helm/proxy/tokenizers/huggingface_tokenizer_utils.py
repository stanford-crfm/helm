import os
from typing import Dict
from threading import Lock

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from helm.common.hierarchical_logger import htrack_block, hlog


class HuggingFacePreTrainedTokenizerFactory:
    """A factory that creates and caches Hugging Face PreTrainedTokenizerBase tokenizers."""

    _tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
    _tokenizers_lock: Lock = Lock()

    @staticmethod
    def create_tokenizer(pretrained_model_name_or_path: str, **kwargs) -> PreTrainedTokenizerBase:
        """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
        # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
        # TODO: Figure out if we actually need this.
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        with HuggingFacePreTrainedTokenizerFactory._tokenizers_lock:
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
                    pretrained_model_name_or_path, local_files_only=True, use_fast=True, **kwargs
                )
            except OSError:
                hlog(
                    f"Local files do not exist for HuggingFace tokenizer: "
                    f"{pretrained_model_name_or_path}. Downloading..."
                )
                return AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path, local_files_only=False, use_fast=True, **kwargs
                )

    @staticmethod
    def get_tokenizer(
        helm_tokenizer_name: str, pretrained_model_name_or_path: str, **kwargs
    ) -> PreTrainedTokenizerBase:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        with HuggingFacePreTrainedTokenizerFactory._tokenizers_lock:
            if helm_tokenizer_name not in HuggingFacePreTrainedTokenizerFactory._tokenizers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (kwargs={kwargs}) "
                    f"for HELM tokenizer {helm_tokenizer_name} with Hugging Face Transformers"
                ):

                    # Keep the tokenizer in memory, so we don't recreate it for future requests
                    HuggingFacePreTrainedTokenizerFactory._tokenizers[
                        helm_tokenizer_name
                    ] = HuggingFacePreTrainedTokenizerFactory.create_tokenizer(pretrained_model_name_or_path, **kwargs)

        return HuggingFacePreTrainedTokenizerFactory._tokenizers[helm_tokenizer_name]
