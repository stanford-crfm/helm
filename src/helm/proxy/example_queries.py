import textwrap

from helm.proxy.query import Query


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
            model: openai/gpt-3.5-turbo-0613
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
            num_completions: 5  # Generate many samples
            model: openai/gpt-3.5-turbo-0613
            """
        ),
        environments="",
    ),
    # Disabled because `max_tokens: 0` no longer works on the OpenAI API
    # Query(
    #     prompt="The quick brown fox jumps over the lazy dog.",
    #     settings=dedent(
    #         """
    #         echo_prompt: true  # Analyze the prompt
    #         max_tokens: 0  # Don't generate any more
    #         top_k_per_token: 5  # Show alternatives for each position
    #         model: openai/text-davinci-002
    #         model_deployment: openai/text-davinci-002
    #         """
    #     ),
    #     environments=dedent(""),
    # ),
    Query(
        prompt="Odd numbers: 1 -> 3 -> 5",
        settings=dedent(
            """
            temperature: 0  # Deterministic
            max_tokens: 50
            model: openai/gpt-3.5-turbo-0613
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
            # Try out multiple models
            model: ${model}
            """
        ),
        environments=dedent(
            """
            occupation: [mathematician, lawyer, doctor]
            model: [openai/gpt-3.5-turbo-0613, openai/gpt-3.5-turbo-1106]
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
            # Try out multiple models
            model: ${model}
            """
        ),
        environments=dedent(
            """
            model: [openai/gpt-3.5-turbo-0613, openai/gpt-3.5-turbo-1106]
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
            # Try out multiple models
            model: ${model}
            """
        ),
        environments=dedent(
            """
            model: [openai/gpt-3.5-turbo-0613, openai/gpt-3.5-turbo-1106]
            """
        ),
    ),
    Query(
        prompt="Write a Python function that takes two vectors a and b and returns their Euclidean distance.",
        settings=dedent(
            """
            model: openai/gpt-3.5-turbo-0613
            """
        ),
        environments="",
    ),
    Query(
        prompt="The quick brown fox",
        settings=dedent(
            """
            temperature: 0.3
            stop_sequences: [\\n]
            # Try out multiple models
            model: ${model}
            """
        ),
        environments=dedent(
            """
            model: [openai/gpt-3.5-turbo-0613, openai/gpt-3.5-turbo-1106]
            """
        ),
    ),
]
