from .client import truncate_sequence
from typing import List
from helm.common.request import Request, Sequence, Token


def truncate_sequence_helper(tokens: List[str], request: Request, expected_tokens: List[str]):
    sequence = Sequence(
        text="".join(tokens),
        tokens=[Token(text=text, logprob=-1, top_logprobs={}) for text in tokens],
        logprob=-len(tokens),
    )

    output_sequence = truncate_sequence(sequence, request)

    assert expected_tokens == [token.text for token in output_sequence.tokens]
    assert "".join(expected_tokens) == output_sequence.text
    assert output_sequence.logprob == sum(token.logprob for token in output_sequence.tokens)


def test_truncate_sequence():
    # echo_prompt = True, nothing gets truncated
    truncate_sequence_helper(["a", "b", "c"], Request(prompt="abc", echo_prompt=True), ["a", "b", "c"])

    # Nothing gets truncated
    truncate_sequence_helper(["hello", " world"], Request(stop_sequences=["#"]), ["hello", " world"])

    # Truncate using stop sequences
    truncate_sequence_helper(["hello", " world", "\n", "what"], Request(stop_sequences=["\n"]), ["hello", " world"])

    # Truncate using max tokens
    truncate_sequence_helper(["a", "b", "c"], Request(max_tokens=2), ["a", "b"])
