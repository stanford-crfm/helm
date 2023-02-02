from .models import get_model, get_model_group, get_models_by_organization, get_all_code_models, Model


def test_get_model():
    model: Model = get_model("ai21/j1-jumbo")
    assert model.organization == "ai21"
    assert model.engine == "j1-jumbo"


def test_get_model_with_invalid_model_name():
    try:
        get_model("invalid/model")
        assert False, "Expected to throw ValueError"
    except ValueError:
        pass


def test_get_model_group():
    assert get_model_group("openai/text-curie-001") == "gpt3"


def test_get_models_by_organization():
    assert get_models_by_organization("simple") == ["simple/model1"]


def test_all_code_models():
    assert "openai/code-davinci-002" in get_all_code_models()
