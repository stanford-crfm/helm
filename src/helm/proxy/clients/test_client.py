from .client import truncate_completion
from typing import List
from helm.common.request import Request, GeneratedOutput, Token


def truncate_completion_helper(tokens: List[str], request: Request, expected_tokens: List[str]):
    completion = GeneratedOutput(
        text="".join(tokens),
        tokens=[Token(text=text, logprob=-1) for text in tokens],
        logprob=-len(tokens),
    )

    output_completion = truncate_completion(completion, request)

    assert expected_tokens == [token.text for token in output_completion.tokens]
    assert "".join(expected_tokens) == output_completion.text
    assert output_completion.logprob == sum(token.logprob for token in output_completion.tokens)


def test_truncate_completion():
    # echo_prompt = True, nothing gets truncated
    truncate_completion_helper(
        ["a", "b", "c"],
        Request(
            model="openai/text-davinci-002", model_deployment="openai/text-davinci-002", prompt="abc", echo_prompt=True
        ),
        ["a", "b", "c"],
    )

    # Nothing gets truncated
    truncate_completion_helper(
        ["hello", " world"],
        Request(model="openai/text-davinci-002", model_deployment="openai/text-davinci-002", stop_sequences=["#"]),
        ["hello", " world"],
    )

    # Truncate using stop sequences
    truncate_completion_helper(
        ["hello", " world", "\n", "what"],
        Request(model="openai/text-davinci-002", model_deployment="openai/text-davinci-002", stop_sequences=["\n"]),
        ["hello", " world"],
    )

    # Truncate using max tokens
    truncate_completion_helper(
        ["a", "b", "c"],
        Request(model="openai/text-davinci-002", model_deployment="openai/text-davinci-002", max_tokens=2),
        ["a", "b"],
    )
