import os
from typing import Dict, Optional
from threading import Lock

from transformers import AutoTokenizer

from helm.common.hierarchical_logger import htrack_block, hlog


# TODO: Rename HuggingFaceTokenizer to disambiguate Hugging Face AutoTokenizers and HELM Tokenizers.
# TODO: Rename HuggingFaceTokenizers to a Factory.
class HuggingFaceTokenizers:
    """A factory that creates and caches Hugging Face tokenizers."""

    _tokenizers: Dict[str, AutoTokenizer] = {}
    _tokenizers_lock: Lock = Lock()

    @staticmethod
    def create_tokenizer(pretrained_model_name_or_path: str, revision: Optional[str] = None) -> AutoTokenizer:
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
    ) -> AutoTokenizer:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        with HuggingFaceTokenizers._tokenizers_lock:
            if helm_tokenizer_name not in HuggingFaceTokenizers._tokenizers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (revision={revision}) "
                    f"for HELM tokenizer {helm_tokenizer_name} with Hugging Face Transformers"
                ):

                    # Keep the tokenizer in memory, so we don't recreate it for future requests
                    HuggingFaceTokenizers._tokenizers[helm_tokenizer_name] = HuggingFaceTokenizers.create_tokenizer(
                        pretrained_model_name_or_path, revision
                    )

        return HuggingFaceTokenizers._tokenizers[helm_tokenizer_name]
