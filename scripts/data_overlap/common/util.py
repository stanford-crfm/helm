from light_tokenizer import LightTokenizer, DefaultTokenizer


def get_tokenizer(normalization) -> LightTokenizer:
    if normalization == "none":
        return LightTokenizer()
    elif normalization == "default":
        return DefaultTokenizer()
    else:
        raise ValueError(f"Normalization strategy {normalization} is not defined.")
