import os
from typing import Any, Dict

from transformers import GPT2TokenizerFast

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
                return load_method(hf_tokenizer_name, local_files_only=True)
            except OSError:
                hlog(f"Local files do not exist for HuggingFace tokenizer: {hf_tokenizer_name}. Downloading...")
                return load_method(hf_tokenizer_name, local_files_only=False)

        if tokenizer_name not in HuggingFaceTokenizers.tokenizers:
            with htrack_block(f"Loading {tokenizer_name} with Hugging Face Transformers"):
                # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
                os.environ["TOKENIZERS_PARALLELISM"] = "False"

                # Weights are cached at ~/.cache/huggingface/transformers.
                if tokenizer_name == "huggingface/gpt2_tokenizer_fast":
                    tokenizer = load_tokenizer(GPT2TokenizerFast.from_pretrained, "gpt2")
                else:
                    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

                # Cache tokenizer in memory
                HuggingFaceTokenizers.tokenizers[tokenizer_name] = tokenizer

        return HuggingFaceTokenizers.tokenizers[tokenizer_name]
