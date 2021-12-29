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
            temperature: 0.5
            maxTokens: 100
            #model: ai21/j1-jumbo
            model: openai/davinci
            """
        ),
        environments="",
    ),
    Query(
        prompt="A ${occupation} is someone who",
        settings=dedent(
            """
            temperature: 0.5
            maxTokens: 100
            model: ${model}
            """
        ),
        environments=dedent(
            """
            occupation: [mathematician, lawyer, doctor, programmer, president]
            model: [ai21/j1-jumbo, openai/davinci]
            """
        ),
    ),
    Query(
        prompt='"""Takes two vectors a and b and returns their Euclidean distance"""',
        settings=dedent(
            """
            model: openai/davinci-codex
            """
        ),
        environments="",
    ),
]
