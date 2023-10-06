# mypy: check_untyped_defs = False
from typing import Any, Dict, Optional
import os

from transformers import AutoTokenizer

from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)
from .cachable_tokenizer import CachableTokenizer
from .tokenizer import cleanup_tokens


# Map of HELM tokenizer name to Hugging Face Hub tokenizer name where they differ.
_KNOWN_TOKENIZER_ALIASES: Dict[str, str] = {
    "huggingface/gpt2": "gpt2",
    "google/t5-11b": "t5-11b",
}


class HuggingFaceTokenizer(CachableTokenizer):
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

        if tokenizer_name not in HuggingFaceTokenizer.tokenizers:
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
                elif tokenizer_name in _KNOWN_TOKENIZER_ALIASES:
                    hf_tokenizer_name = _KNOWN_TOKENIZER_ALIASES[tokenizer_name]
                else:
                    hf_tokenizer_name = tokenizer_name

                # Keep the tokenizer in memory, so we don't recreate it for future requests
                HuggingFaceTokenizer.tokenizers[tokenizer_name] = load_tokenizer(hf_tokenizer_name, revision)

        return HuggingFaceTokenizer.tokenizers[tokenizer_name]

    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        _tokenizer = HuggingFaceTokenizer.get_tokenizer(request.tokenizer)
        if request.encode:
            if request.truncation:
                tokens = _tokenizer.encode(
                    request.text,
                    truncation=request.truncation,
                    max_length=request.max_length,
                    add_special_tokens=False,
                )
            else:
                tokens = _tokenizer.encode(request.text, add_special_tokens=False)
            return {"token_ids": tokens}
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
                tokens = [_tokenizer.convert_tokens_to_string([token]) for token in _tokenizer.tokenize(request.text)]
            else:
                # Tokenizes the text and returns the tokens as a list of strings,
                # not a list of token objects (otherwise "Hello world" would be"
                # ["Hello", "▁world"] and not ["Hello", " world"])
                # We could do this with a simple replace like this:
                # tokens = [_tokenizer.convert_tokens_to_string([i]) for i in _tokenizer.tokenize(request.text)]
                # But this replaces all the "▁" characters by "", which is not what we want.
                # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                # Just like tokenize("Hello", encode=False) would return ["Hello"].
                tokens = _tokenizer.tokenize(request.text)
                tokens = cleanup_tokens(tokens, request.tokenizer)
            return {"token_strings": tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        _tokenizer = HuggingFaceTokenizer.get_tokenizer(request.tokenizer)
        text = _tokenizer.decode(request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces)
        return {"text": text}
