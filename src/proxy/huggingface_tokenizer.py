import os
from typing import Any, Dict

from transformers import GPT2TokenizerFast, AutoTokenizer, GPTNeoXTokenizerFast

from common.hierarchical_logger import htrack_block, hlog


class HuggingFaceTokenizers:

    tokenizers: Dict[str, Any] = {}

    @staticmethod
    def get_tokenizer(tokenizer_name: str) -> Any:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """

        def load_tokenizer(load_method: Any, hf_tokenizer_name: str):
            """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
            try:
                # From the HuggingFace documentation, "local_files_only(defaults to False) —
                # Whether or not to only look at local files".
                # Running `local_files_only=False` requires an internet connection even if the files are downloaded
                # and cached. We need to first run with `local_files_only=True` just in case the machine
                # we are running this code has connection issues. If the tokenizer files are not cached,
                # we attempt to download them from HuggingFace.
                return load_method(hf_tokenizer_name, local_files_only=True)
            except OSError:
                hlog(f"Local files do not exist for HuggingFace tokenizer: {hf_tokenizer_name}. Downloading...")
                return load_method(hf_tokenizer_name, local_files_only=False)

        if tokenizer_name not in HuggingFaceTokenizers.tokenizers:
            with htrack_block(f"Loading {tokenizer_name} with Hugging Face Transformers"):
                # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
                os.environ["TOKENIZERS_PARALLELISM"] = "False"

                # Use the "fast" version of the tokenizer if available.
                # Weights are cached at ~/.cache/huggingface/transformers.
                if tokenizer_name == "huggingface/gpt2":
                    tokenizer = load_tokenizer(GPT2TokenizerFast.from_pretrained, "gpt2")
                elif tokenizer_name == "huggingface/gpt-j-6b":
                    # Not a typo: Named "gpt-j-6B" instead of "gpt-j-6b" in HuggingFace
                    tokenizer = load_tokenizer(AutoTokenizer.from_pretrained, "EleutherAI/gpt-j-6B")
                elif tokenizer_name == "huggingface/gpt-neox-20b":
                    tokenizer = load_tokenizer(GPTNeoXTokenizerFast.from_pretrained, "EleutherAI/gpt-neox-20b")
                else:
                    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

                # Keep the tokenizer in memory, so we don't recreate it for future requests
                HuggingFaceTokenizers.tokenizers[tokenizer_name] = tokenizer

        return HuggingFaceTokenizers.tokenizers[tokenizer_name]
