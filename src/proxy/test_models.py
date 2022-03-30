from .models import get_model_group, get_all_code_models, get_all_text_models


def test_get_model_group():
    assert get_model_group("openai/text-curie-001") == "gpt3"


def test_all_text_models():
    assert get_all_text_models() == [
        "ai21/j1-jumbo",
        "ai21/j1-large",
        "openai/davinci",
        "openai/curie",
        "openai/babbage",
        "openai/ada",
        "openai/text-davinci-001",
        "openai/text-curie-001",
        "openai/text-babbage-001",
        "openai/text-ada-001",
        "huggingface/gptj_6b",
        "huggingface/gpt2",
    ]


def test_all_code_models():
    assert get_all_code_models() == ["openai/code-davinci-001", "openai/code-cushman-001"]
