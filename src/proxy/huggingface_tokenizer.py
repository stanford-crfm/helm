import os
from typing import Any, Dict

from transformers import GPT2TokenizerFast, AutoTokenizer, GPTNeoXTokenizerFast

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

                # Use the "fast" version of the tokenizer if available.
                # Weights are cached at ~/.cache/huggingface/transformers.
                if tokenizer_name == "huggingface/gpt2-tokenizer-fast":
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                elif tokenizer_name == "huggingface/gpt-j-6b":
                    # Not a typo: Named "gpt-j-6B" instead of "gpt-j-6b" in HuggingFace
                    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
                elif tokenizer_name == "huggingface/gpt-neox-20b":
                    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
                else:
                    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

                # Keep the tokenizer in memory, so we don't recreate it for future requests
                HuggingFaceTokenizers.tokenizers[tokenizer_name] = tokenizer

        return HuggingFaceTokenizers.tokenizers[tokenizer_name]
