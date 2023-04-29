import pytest
import unittest
from typing import List, Tuple

from helm.benchmark.run_expander import ModelRunExpander
from helm.proxy.clients.huggingface_model_registry import (
    HuggingFaceModelConfig,
    register_huggingface_model_config,
    get_huggingface_model_config,
)
from helm.proxy.models import get_all_models, get_all_text_models


@pytest.mark.parametrize("model_name", ["EleutherAI/pythia-70m"])
def test_hf_model_register(model_name):
    register_huggingface_model_config(model_name)
    assert model_name in ModelRunExpander("all").values
    assert model_name in get_all_models()
    assert model_name in get_all_text_models()


class TestHuggingFaceModelRegistry(unittest.TestCase):
    def test_round_trip(self):
        config_pairs: List[Tuple[str, HuggingFaceModelConfig]] = [
            ("gpt2", HuggingFaceModelConfig(namespace=None, model_name="gpt2", revision=None)),
            (
                "stanford-crfm/BioMedLM",
                HuggingFaceModelConfig(namespace="stanford-crfm", model_name="BioMedLM", revision=None),
            ),
            (
                "stanford-crfm/BioMedLM@main",
                HuggingFaceModelConfig(namespace="stanford-crfm", model_name="BioMedLM", revision="main"),
            ),
        ]
        for expected_model_name, expected_model_config in config_pairs:
            actual_model_config = HuggingFaceModelConfig.from_string(expected_model_name)
            actual_model_name = str(actual_model_config)
            self.assertEqual(actual_model_name, expected_model_name)
            self.assertEqual(actual_model_config, expected_model_config)

    def test_model_id(self):
        config_pairs: List[Tuple[str, str]] = [
            ("gpt2", "gpt2"),
            ("stanford-crfm/BioMedLM", "stanford-crfm/BioMedLM"),
            ("stanford-crfm/BioMedLM@main", "stanford-crfm/BioMedLM"),
        ]
        for expected_model_name, expected_model_id in config_pairs:
            actual_model_config = HuggingFaceModelConfig.from_string(expected_model_name)
            self.assertEqual(actual_model_config.model_id, expected_model_id)

    def test_register_huggingface_model_config(self):
        register_huggingface_model_config("stanford-crfm/BioMedLM@main")
        expected_model_config = HuggingFaceModelConfig(
            namespace="stanford-crfm", model_name="BioMedLM", revision="main"
        )
        actual_model_config = get_huggingface_model_config("stanford-crfm/BioMedLM@main")
        self.assertEqual(actual_model_config, expected_model_config)
