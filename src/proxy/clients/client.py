import time
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, List

from common.hierarchical_logger import hlog
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)


class Client(ABC):
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

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        pass

    @abstractmethod
    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        pass

    @abstractmethod
    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        pass


def wrap_request_time(compute: Callable[[], Any]) -> Callable[[], Any]:
    """Return a version of `compute` that puts `request_time` into its output."""

    def wrapped_compute():
        start_time = time.time()
        response = compute()
        end_time = time.time()
        response["request_time"] = end_time - start_time
        response["request_datetime"] = int(start_time)
        return response

    return wrapped_compute


def truncate_sequence(sequence: Sequence, request: Request, print_warning: bool = True) -> Sequence:
    """
    Certain providers have bugs where they aren't respecting max_tokens or
    stop_sequences, so as a hack, we have to manually truncate the suffix of
    `sequence` as a post-hoc process.
    """
    for stop in request.stop_sequences:
        # Find `stop` in the text
        try:
            new_text = sequence.text[: sequence.text.index(stop)]
        except ValueError:
            # Stop sequence doesn't exist, skip
            continue

        # At this point, we've stripped out `stop`...

        # Strip `stop` off the tokens
        new_tokens: List[Token] = []
        for token in sequence.tokens:
            # Note: we can only strip at token boundaries
            if token.text.startswith(stop):
                break
            new_tokens.append(token)
        if len(new_tokens) == len(sequence.tokens):
            hlog(
                f"WARNING: Stripped characters from text ({len(sequence.text)} -> {len(new_text)}), "
                f"but wasn't able to strip the tokens"
            )

        # Recompute log probability
        new_logprob = sum(token.logprob for token in new_tokens)

        if print_warning:
            hlog(f"WARNING: truncate_sequence needs to strip {stop}")

        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    # Truncate based on the max number of tokens
    if len(sequence.tokens) > request.max_tokens:
        if print_warning:
            hlog(f"WARNING: truncate_sequence needs to truncate {len(sequence.tokens)} down to {request.max_tokens}")
        new_tokens = sequence.tokens[: request.max_tokens]

        # This is imperfect stitching together of tokens, so just to make sure this is okay
        # TODO: should use the proper detokenizer since T5-style models.
        # Usually, in our benchmark max_tokens is active when it's 1, so hopefully this isn't an issue.
        new_text = "".join(token.text for token in new_tokens)
        if not sequence.text.startswith(new_text):
            hlog(f"WARNING: {sequence.text} does not start with truncated text {new_text}")

        new_logprob = sum(token.logprob for token in new_tokens)

        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    return sequence
