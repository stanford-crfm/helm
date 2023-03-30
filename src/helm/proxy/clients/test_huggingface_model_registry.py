import pytest
from .huggingface_model_registry import register_huggingface_model_config
from ..models import get_all_models, get_all_text_models
from ...benchmark.run_expander import ModelRunExpander


@pytest.mark.parametrize("model_name", ["EleutherAI/pythia-70m"])
def test_hf_model_register(model_name):
    register_huggingface_model_config(model_name)
    assert model_name in ModelRunExpander("all").values
    assert model_name in get_all_models()
    assert model_name in get_all_text_models()
