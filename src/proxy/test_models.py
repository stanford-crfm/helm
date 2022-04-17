from .models import get_model_group, get_all_code_models


def test_get_model_group():
    assert get_model_group("openai/text-curie-001") == "gpt3"


def test_all_code_models():
    assert get_all_code_models() == ["openai/code-davinci-001", "openai/code-cushman-001"]
