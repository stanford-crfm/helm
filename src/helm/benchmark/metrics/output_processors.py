import re


def remove_deepseek_r1_thinking(input: str) -> str:
    if "<think>" not in input:
        return input

    if "</think>\n\n" in input:
        # The think block is usually followed by two newlines, so we should remove that
        return re.sub("<think>.*</think>\n\n", "", input, flags=re.DOTALL)
    elif "</think>" in input:
        return re.sub("<think>.*</think>", "", input, flags=re.DOTALL)
    else:
        # Unclosed think block
        return ""
