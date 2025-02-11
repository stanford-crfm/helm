import re


def remove_thinking(input: str) -> str:
    result = input
    result = re.sub("<think>.*</think>", "", result, flags=re.DOTALL)
    # If there is a unclosed think block, remove the entire block
    result = re.sub("<think>.*", "", result, flags=re.DOTALL)
    return result
