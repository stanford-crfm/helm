import json
from abc import ABC, abstractmethod
from typing import List, Mapping, Optional, cast

from helm.common.hierarchical_logger import hlog
from helm.common.media_object import MultimediaObject, TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.cache import Cache, CacheConfig
from helm.common.tokenization_request import DecodeRequest, TokenizationRequest
from helm.tokenizers.tokenizer import Tokenizer


class Client(ABC):
    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        """Makes a request to the model.

        For LLM, this usually corresponds to a single call to the model (completion).
        """
        pass


class CachingClient(Client):
    def __init__(self, cache_config: CacheConfig) -> None:
        """Initializes the client.

        For most clients, both the cache config and tokenizer are required.
        However, some clients, such as the auto client, handle multiple tokenizers,
        and organizations so the cache and tokenizer cannot be initialized until
        the request is made.
        """
        self.cache = Cache(cache_config) if cache_config is not None else None

    @staticmethod
    def make_cache_key(raw_request: Mapping, request: Request) -> Mapping:
        """
        Construct the key for the cache using the raw request.
        Add `request.random` to the key, if defined.
        """
        if request.random is not None:
            assert "random" not in raw_request
            return {**raw_request, "random": request.random}
        else:
            return {**raw_request}


def truncate_sequence(
    sequence: GeneratedOutput,
    request: Request,
    end_of_text_token: Optional[str] = None,
    print_warning: bool = True,
) -> GeneratedOutput:
    """
    Certain providers have bugs where they aren't respecting max_tokens,
    stop_sequences and the end of text token, so as a hack, we have to manually
    truncate the suffix of `sequence` and `tokens` as a post-hoc process.

    This method is unsafe and may produce warnings or incorrect results.
    Prefer using the safer truncate_and_tokenize_response_text() method instead
    if your use case satisfies its requirements.
    """
    # TODO: if echo_prompt, then we should only ignore the prompt, but we don't
    # know how many tokens the prompt takes up.
    # In the benchmark, usually echo_prompt is only used for language modeling,
    # where max_tokens = 0, so there's nothing to truncate.
    if request.echo_prompt:
        if request.max_tokens != 0:
            hlog("WARNING: don't know how to handle echo_prompt and max_tokens > 0, not truncating")
        return sequence

    if end_of_text_token:
        stop_sequences = request.stop_sequences + [end_of_text_token]
    else:
        stop_sequences = request.stop_sequences
    for stop in stop_sequences:
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

        sequence = GeneratedOutput(text=new_text, logprob=new_logprob, tokens=new_tokens)

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

        sequence = GeneratedOutput(text=new_text, logprob=new_logprob, tokens=new_tokens)

    return sequence


def truncate_and_tokenize_response_text(
    text: str,
    request: Request,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    end_of_text_token: Optional[str] = None,
    original_finish_reason: str = "endoftext",
) -> GeneratedOutput:
    """Truncate a string-only response to respect stop_sequences and max_tokens.

    This can only be used if all of the following conditions are true:

    - You have access to the tokenizer.
    - The request has echo_prompt = False.
    - The tokenizer supports encoding and decoding.
    - The tokenizer's tokenize() method supports truncation.
    - The model's response is text-only.
    - The model's response not already provide the tokenized text.
    - The model's response does not provide logprobs.

    This method is safer than truncate_sequence() and should be preferred if the above conditions are met.
    Unlike truncate_sequence(), this method will not produce warnings or incorrect results.
    This is because the the tokens are derived from the truncated text using the tokenizer,
    so the text and the tokens in the resulting result are guranteed to match."""
    # Finish reason strings are token from basic_metrics._compute_finish_reason_metrics()
    finish_reason: str = original_finish_reason
    if request.echo_prompt:
        raise Exception("truncate_and_tokenize_response_text() does not support requests with echo_prompt = True")

    if end_of_text_token:
        stop_sequences = request.stop_sequences + [end_of_text_token]
    else:
        stop_sequences = request.stop_sequences
    for stop_sequence in stop_sequences:
        try:
            text = text[: text.index(stop_sequence)]
            finish_reason = "stop"
        except ValueError:
            pass

    token_strings = cast(
        List[str], tokenizer.tokenize(TokenizationRequest(text=text, tokenizer=tokenizer_name)).raw_tokens
    )
    if len(token_strings) > request.max_tokens:
        encoded_ints = cast(
            List[int],
            tokenizer.tokenize(
                TokenizationRequest(
                    text=text, tokenizer=tokenizer_name, encode=True, truncation=True, max_length=request.max_tokens
                )
            ).raw_tokens,
        )
        text = tokenizer.decode(DecodeRequest(encoded_ints, tokenizer_name)).text
        token_strings = cast(
            List[str], tokenizer.tokenize(TokenizationRequest(text=text, tokenizer=tokenizer_name)).raw_tokens
        )
        finish_reason = "length"
    tokens = [Token(text=token_string, logprob=0.0) for token_string in token_strings]
    return GeneratedOutput(text=text, logprob=0.0, tokens=tokens, finish_reason={"reason": finish_reason})


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
    elif tokenizer_name is not None and (tokenizer_name.startswith("huggingface") or tokenizer_name.endswith("gpt2")):
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
