import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from helm.common.hierarchical_logger import hlog
from helm.common.media_object import MultimediaObject, TEXT_TYPE
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.common.cache import Cache, CacheConfig
from helm.proxy.tokenizers.tokenizer import Tokenizer


class Client(ABC):
    # TODO: This method should be removed.
    # This only kept for the AutoClient. Eventually, we should introduce an
    # AutoTokenizer or TokenizerFactory class.
    @abstractmethod
    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes `request.text` using `request.tokenizer`.

        This simply calls the `tokenize` method of the tokenizer.
        Some exceptions can be made (but should be avoided).
        This is the case for the auto client, which needs to handle
        tokenization for multiple tokenizers.
        """
        pass

    # TODO: This method should be removed.
    # This only kept for the AutoClient. Eventually, we should introduce an
    # AutoTokenizer or TokenizerFactory class.
    @abstractmethod
    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes `request.tokens` using `request.tokenizer`.

        This simply calls the `decode` method of the tokenizer.
        Some exceptions can be made (but should be avoided).
        This is the case for the auto client, which needs to handle
        tokenization for multiple tokenizers.
        """
        pass

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        """Makes a request to the model.

        For LLM, this usually corresponds to a single call to the model (completion).
        """
        pass


class CachingClient(Client):
    def __init__(self, cache_config: CacheConfig, tokenizer: Tokenizer) -> None:
        """Initializes the client.

        For most clients, both the cache config and tokenizer are required.
        However, some clients, such as the auto client, handle multiple tokenizers,
        and organizations so the cache and tokenizer cannot be initialized until
        the request is made.
        """
        self.cache = Cache(cache_config) if cache_config is not None else None
        self.tokenizer = tokenizer

    @staticmethod
    def make_cache_key(raw_request: Dict, request: Request) -> Dict:
        """
        Construct the key for the cache using the raw request.
        Add `request.random` to the key, if defined.
        """
        if request.random is not None:
            assert "random" not in raw_request
            cache_key = {**raw_request, "random": request.random}
        else:
            cache_key = raw_request
        return cache_key

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        # Deprecated - use `self.tokenizer.tokenize` instead. Warn the user.
        hlog("WARNING: CachingClient.tokenize is deprecated, use self.tokenizer.tokenize instead")
        return self.tokenizer.tokenize(request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        # Deprecated - use `self.tokenizer.decode` instead. Warn the user.
        hlog("WARNING: CachingClient.decode is deprecated, use self.tokenizer.decode instead")
        return self.tokenizer.decode(request)


def truncate_sequence(sequence: Sequence, request: Request, print_warning: bool = True) -> Sequence:
    """
    Certain providers have bugs where they aren't respecting max_tokens,
    stop_sequences and the end of text token, so as a hack, we have to manually
    truncate the suffix of `sequence` and `tokens` as a post-hoc process.
    """
    # TODO: if echo_prompt, then we should only ignore the prompt, but we don't
    # know how many tokens the prompt takes up.
    # In the benchmark, usually echo_prompt is only used for language modeling,
    # where max_tokens = 0, so there's nothing to truncate.
    if request.echo_prompt:
        if request.max_tokens != 0:
            hlog("WARNING: don't know how to handle echo_prompt and max_tokens > 0, not truncating")
        return sequence

    for stop in request.stop_sequences:
        # Find `stop` in the text
        try:
            new_text = sequence.text[: sequence.text.index(stop)]
        except ValueError:
            # The stop sequence doesn't exist, but it might exist in the list of tokens.
            new_text = sequence.text

        # Strip `stop` off the tokens
        new_tokens: List[Token] = []
        # Need to start
        for token in sequence.tokens:
            # Note: we can only strip at token boundaries
            if token.text.startswith(stop):
                break
            new_tokens.append(token)

        if len(new_text) < len(sequence.text) and len(new_tokens) == len(sequence.tokens):
            hlog(
                f"WARNING: Stripped characters from text ({len(sequence.text)} -> {len(new_text)}), "
                f"but wasn't able to strip the tokens"
            )

        # Recompute log probability
        new_logprob = sum(token.logprob for token in new_tokens)

        if print_warning:
            hlog(f"WARNING: truncate_sequence needs to strip {json.dumps(stop)}")

        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    # Truncate based on the max number of tokens.
    if len(sequence.tokens) > request.max_tokens:
        if print_warning:
            hlog(f"WARNING: truncate_sequence needs to truncate {len(sequence.tokens)} down to {request.max_tokens}")
        new_tokens = sequence.tokens[: request.max_tokens]

        # This is imperfect stitching together of tokens, so just to make sure this is okay
        # TODO: should use the proper detokenizer since T5-style models.
        # Usually, in our benchmark, max_tokens is active when it's 1, so hopefully this isn't an issue.
        new_text = "".join(token.text for token in new_tokens)
        if not sequence.text.startswith(new_text):
            hlog(f"WARNING: {json.dumps(sequence.text)} does not start with truncated text {json.dumps(new_text)}")

        new_logprob = sum(token.logprob for token in new_tokens)

        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    return sequence


def cleanup_str(token: str, tokenizer_name: Optional[str] = None) -> str:
    """
    Certain tokenizers introduce special characters to represent spaces, such as
    "Ġ" or "▁". This function removes those characters.
    """
    if tokenizer_name in [
        "TsinghuaKEG/ice",
        "bigscience/T0pp",
        "google/t5-11b",
        "google/flan-t5-xxl",
        "google/ul2",
        "Yandex/yalm",
        "ai21/j1",
        "together",
    ]:
        return token.replace("▁", " ")
    elif tokenizer_name is not None and tokenizer_name.startswith("huggingface"):
        return token.replace("Ġ", " ")
    return token


def cleanup_tokens(tokens: List[str], tokenizer_name: Optional[str] = None) -> List[str]:
    """
    Applies `cleanup_str` to each token in `tokens`.
    """
    return [cleanup_str(token, tokenizer_name) for token in tokens]


def generate_uid_for_multimodal_prompt(prompt: MultimediaObject) -> str:
    """Generates a unique identifier for a given multimodal prompt."""
    return "".join(
        [
            media_object.text if media_object.is_type(TEXT_TYPE) and media_object.text else str(media_object.location)
            for media_object in prompt.media_objects
        ]
    )
