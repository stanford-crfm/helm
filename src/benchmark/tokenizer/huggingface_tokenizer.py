import os
from typing import Any, Dict

from transformers import GPT2TokenizerFast

from common.hierarchical_logger import htrack_block


class HuggingFaceTokenizers:

    tokenizers: Dict[str, Any] = {}

    @staticmethod
    def get_tokenizer(tokenizer_name: str) -> Any:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        if tokenizer_name not in HuggingFaceTokenizers.tokenizers:
            with htrack_block(f"Creating {tokenizer_name} with Hugging Face Transformers"):
                # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
                os.environ["TOKENIZERS_PARALLELISM"] = "False"

                # Weights are cached at ~/.cache/huggingface/transformers.
                if tokenizer_name == "huggingface/gpt2_tokenizer_fast":
                    HuggingFaceTokenizers.tokenizers[tokenizer_name] = GPT2TokenizerFast.from_pretrained("gpt2")
                else:
                    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

        return HuggingFaceTokenizers.tokenizers[tokenizer_name]
