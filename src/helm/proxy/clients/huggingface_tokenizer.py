import os
from typing import Any, Dict, Optional, Set

from transformers import AutoTokenizer

from helm.common.hierarchical_logger import htrack_block, hlog

from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)


# Tokenizer names where the HELM tokenizer name and the Hugging Face tokenizer name
# are identical.
_KNOWN_TOKENIZER_NAMES: Set[str] = {
    "EleutherAI/gpt-j-6B",  # Not a typo: Named "gpt-j-6B" instead of "gpt-j-6b" in Hugging Face
    "EleutherAI/gpt-neox-20b",
    "bigscience/bloom",
    "bigscience/T0pp",
    "facebook/opt-66b",
    "google/ul2",
    "google/flan-t5-xxl",
    "meta-llama/Llama-2-7b-hf",
    "bigcode/santacoder",
    "bigcode/starcoder",
    "hf-internal-testing/llama-tokenizer",
    "openai/clip-vit-large-patch14",
}


# Map of HELM tokenizer name to Hugging Face tokenizer name for tokenizers where they differ.
_KNOWN_TOKENIZER_ALIASES: Dict[str, str] = {
    "huggingface/gpt2": "gpt2",
    "google/t5-11b": "t5-11b",
}


class HuggingFaceTokenizers:
    tokenizers: Dict[str, Any] = {}

    @staticmethod
    def get_tokenizer(tokenizer_name: str) -> Any:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """

        def load_tokenizer(hf_tokenizer_name: str, revision: Optional[str] = None):
            """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
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
                    hf_tokenizer_name, local_files_only=True, use_fast=True, **tokenizer_kwargs
                )
            except OSError:
                hlog(f"Local files do not exist for HuggingFace tokenizer: {hf_tokenizer_name}. Downloading...")
                return AutoTokenizer.from_pretrained(
                    hf_tokenizer_name, local_files_only=False, use_fast=True, **tokenizer_kwargs
                )

        if tokenizer_name not in HuggingFaceTokenizers.tokenizers:
            with htrack_block(f"Loading {tokenizer_name} with Hugging Face Transformers"):
                # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
                os.environ["TOKENIZERS_PARALLELISM"] = "False"

                # Weights are cached at ~/.cache/huggingface/transformers.
                hf_tokenizer_name: str
                revision: Optional[str] = None
                model_config = get_huggingface_model_config(tokenizer_name)
                if model_config:
                    if isinstance(model_config, HuggingFaceLocalModelConfig):
                        hlog(f"Loading {tokenizer_name} from local path {model_config.path}")
                        hf_tokenizer_name = model_config.path
                        revision = None
                    elif isinstance(model_config, HuggingFaceHubModelConfig):
                        hlog(f"Loading {tokenizer_name} from Hugging Face Hub model {model_config.model_id}")
                        hf_tokenizer_name = model_config.model_id
                        revision = model_config.revision
                    else:
                        raise ValueError(f"Unrecognized Hugging Face model config: {type(model_config)})")
                elif tokenizer_name in _KNOWN_TOKENIZER_NAMES:
                    hf_tokenizer_name = tokenizer_name
                elif tokenizer_name in _KNOWN_TOKENIZER_ALIASES:
                    hf_tokenizer_name = _KNOWN_TOKENIZER_ALIASES[tokenizer_name]
                else:
                    raise ValueError(f"Unsupported HuggingFace tokenizer: {tokenizer_name}")

                # Keep the tokenizer in memory, so we don't recreate it for future requests
                HuggingFaceTokenizers.tokenizers[tokenizer_name] = load_tokenizer(hf_tokenizer_name, revision)

        return HuggingFaceTokenizers.tokenizers[tokenizer_name]
