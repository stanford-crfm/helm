import textwrap

from .query import Query


def dedent(text: str) -> str:
    # Remove leading newline
    if text.startswith("\n"):
        text = text[1:]
    text = textwrap.dedent(text)
    # Remove trailing new line
    if text.endswith("\n"):
        text = text[:-1]
    return text


example_queries = [
    Query(
        prompt="Life is like",
        settings=dedent(
            """
            temperature: 0.5  # Medium amount of randomness
            stop_sequences: [.]  # Stop when you hit a period
            """
        ),
        environments="",
    ),
    Query(
        prompt="Elephants are one of the most",
        settings=dedent(
            """
            temperature: 0.5  # Medium amount of randomness
            stop_sequences: [\\n]  # Stop when you hit a newline
            num_completions: 10  # Generate many samples
            """
        ),
        environments="",
    ),
    Query(
        prompt="The quick brown fox jumps over the lazy dog.",
        settings=dedent(
            """
            echo_prompt: true  # Analyze the prompt
            max_tokens: 0  # Don't generate any more
            top_k_per_token: 10  # Show alternatives for each position
            """
        ),
        environments=dedent(""),
    ),
    Query(
        prompt="Odd numbers: 1 -> 3 -> 5",
        settings=dedent(
            """
            temperature: 0  # Deterministic
            max_tokens: 50
            """
        ),
        environments="",
    ),
    Query(
        prompt="A ${occupation} is someone who",
        settings=dedent(
            """
            temperature: 0
            stop_sequences: [.]
            model: ${model}  # Try out multiple models
            """
        ),
        environments=dedent(
            """
            occupation: [mathematician, lawyer, doctor]
            model: [openai/davinci, ai21/j1-jumbo]
            """
        ),
    ),
    Query(
        prompt=dedent(
            """
            France => Paris
            Germany => Berlin
            China => Beijing
            Japan => Tokyo
            Canada =>
            """
        ),
        settings=dedent(
            """
            temperature: 0.5
            stop_sequences: [\\n]
            num_completions: 5
            model: ${model}  # Try out GPT-3 and Jurassic
            """
        ),
        environments=dedent(
            """
            model: [openai/davinci, ai21/j1-jumbo]
            """
        ),
    ),
    Query(
        prompt=dedent(
            """
            Please answer the following question about geography.

            What is the capital of Germany?
            A. Berlin
            B. Bonn
            C. Hamburg
            D. Munich
            Answer: A

            What is the capital of Canada?
            A. Montreal
            B. Ottawa
            C. Toronto
            D. Vancouver
            Answer:
            """
        ),
        settings=dedent(
            """
            temperature: 0
            max_tokens: 1
            top_k_per_token: 4
            model: ${model}  # Try out GPT-3 and Jurassic
            """
        ),
        environments=dedent(
            """
            model: [openai/davinci, ai21/j1-jumbo]
            """
        ),
    ),
    Query(
        prompt="Takes two vectors a and b and returns their Euclidean distance",
        settings=dedent(
            """
            model: openai/code-davinci-001  # Codex for code generation
            """
        ),
        environments="",
    ),
    Query(
        prompt="The quick brown fox",
        settings=dedent(
            """
            model: ${model}
            temperature: 0.3
            stop_sequences: [\\n]
            """
        ),
        environments=dedent(
            """
            model: [
                "openai/davinci", "openai/text-davinci-002",
                "openai/text-davinci-003", "ai21/j1-grande-v2-beta",
                "together/gpt-j-6b", "together/gpt-jt-6b-v1",
                "together/bloom", "together/opt-175b"
            ]
            """
        ),
    ),
]
