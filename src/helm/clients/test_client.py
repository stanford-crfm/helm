from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.tokenizers.auto_tokenizer import AutoTokenizer
from helm.clients.client import truncate_sequence, truncate_and_tokenize_response_text
from typing import List
from helm.common.request import Request, GeneratedOutput, Token


def truncate_sequence_helper(tokens: List[str], request: Request, expected_tokens: List[str]):
    sequence = GeneratedOutput(
        text="".join(tokens),
        tokens=[Token(text=text, logprob=-1) for text in tokens],
        logprob=-len(tokens),
    )

    output_sequence = truncate_sequence(sequence, request)

    assert expected_tokens == [token.text for token in output_sequence.tokens]
    assert "".join(expected_tokens) == output_sequence.text
    assert output_sequence.logprob == sum(token.logprob for token in output_sequence.tokens)


def test_truncate_sequence():
    # echo_prompt = True, nothing gets truncated
    truncate_sequence_helper(
        ["a", "b", "c"],
        Request(model="openai/gpt2", model_deployment="huggingface/gpt2", prompt="abc", echo_prompt=True),
        ["a", "b", "c"],
    )

    # Nothing gets truncated
    truncate_sequence_helper(
        ["hello", " world"],
        Request(model="openai/gpt2", model_deployment="huggingface/gpt2", stop_sequences=["#"]),
        ["hello", " world"],
    )

    # Truncate using stop sequences
    truncate_sequence_helper(
        ["hello", " world", "\n", "what"],
        Request(model="openai/gpt2", model_deployment="huggingface/gpt2", stop_sequences=["\n"]),
        ["hello", " world"],
    )

    # Truncate using max tokens
    truncate_sequence_helper(
        ["a", "b", "c"],
        Request(model="openai/gpt2", model_deployment="huggingface/gpt2", max_tokens=2),
        ["a", "b"],
    )


def test_truncate_and_tokenize_response_text():
    tokenizer_name = "huggingface/gpt2"
    tokenizer = AutoTokenizer(credentials={}, cache_backend_config=BlackHoleCacheBackendConfig())

    # No truncation
    response = truncate_and_tokenize_response_text(
        "I am a scientist. I am a scientist.", Request(max_tokens=100, stop_sequences=[]), tokenizer, tokenizer_name
    )
    assert response.finish_reason
    assert response.finish_reason["reason"] == "endoftext"
    assert response.text == "I am a scientist. I am a scientist."
    assert response.tokens == [
        Token("I", 0.0),
        Token(" am", 0.0),
        Token(" a", 0.0),
        Token(" scientist", 0.0),
        Token(".", 0.0),
        Token(" I", 0.0),
        Token(" am", 0.0),
        Token(" a", 0.0),
        Token(" scientist", 0.0),
        Token(".", 0.0),
    ]

    response = truncate_and_tokenize_response_text(
        "I am a scientist. I am a scientist.", Request(max_tokens=7, stop_sequences=["."]), tokenizer, tokenizer_name
    )
    assert response.finish_reason
    assert response.finish_reason["reason"] == "stop"
    assert response.text == "I am a scientist"
    assert response.tokens == [Token("I", 0.0), Token(" am", 0.0), Token(" a", 0.0), Token(" scientist", 0.0)]

    response = truncate_and_tokenize_response_text(
        "I am a scientist. I am a scientist.", Request(max_tokens=3, stop_sequences=[]), tokenizer, tokenizer_name
    )
    assert response.finish_reason
    assert response.finish_reason["reason"] == "length"
    assert response.text == "I am a"
    assert response.tokens == [Token("I", 0.0), Token(" am", 0.0), Token(" a", 0.0)]

    response = truncate_and_tokenize_response_text(
        "I am a scientist. I am a scientist.", Request(max_tokens=3, stop_sequences=["."]), tokenizer, tokenizer_name
    )
    assert response.finish_reason
    assert response.finish_reason["reason"] == "length"
    assert response.text == "I am a"
    assert response.tokens == [Token("I", 0.0), Token(" am", 0.0), Token(" a", 0.0)]
