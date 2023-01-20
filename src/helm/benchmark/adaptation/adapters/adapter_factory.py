from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from .adapter import Adapter
from .generation_adapter import GenerationAdapter
from .language_modeling_adapter import LanguageModelingAdapter
from .multiple_choice_joint_adapter import MultipleChoiceJointAdapter
from .multiple_choice_separate_adapter import MultipleChoiceSeparateAdapter
from .multiple_choice_calibrated_adapter import MultipleChoiceCalibratedAdapter
from .binary_ranking_adapter import BinaryRankingAdapter

# Adaptation methods
ADAPT_GENERATION: str = "generation"
ADAPT_LANGUAGE_MODELING: str = "language_modeling"
ADAPT_MULTIPLE_CHOICE_JOINT: str = "multiple_choice_joint"
ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL: str = "multiple_choice_separate_original"
ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED: str = "multiple_choice_separate_calibrated"
ADAPT_RANKING_BINARY: str = "ranking_binary"

ADAPT_MULTIPLE_CHOICE_SEPARATE_METHODS: List[str] = [
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
]


class AdapterFactory:
    @staticmethod
    def get_adapter(adapter_spec: AdapterSpec, tokenizer_service: TokenizerService) -> Adapter:
        """
        Returns a `Adapter` given `method` of `AdapterSpec`.
        """
        method: str = adapter_spec.method
        adapter: Adapter

        if method == ADAPT_GENERATION:
            adapter = GenerationAdapter(adapter_spec, tokenizer_service)
        elif method == ADAPT_LANGUAGE_MODELING:
            adapter = LanguageModelingAdapter(adapter_spec, tokenizer_service)
        elif method == ADAPT_MULTIPLE_CHOICE_JOINT:
            adapter = MultipleChoiceJointAdapter(adapter_spec, tokenizer_service)
        elif method == ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL:
            adapter = MultipleChoiceSeparateAdapter(adapter_spec, tokenizer_service)
        elif method == ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED:
            adapter = MultipleChoiceCalibratedAdapter(adapter_spec, tokenizer_service)
        elif method == ADAPT_RANKING_BINARY:
            adapter = BinaryRankingAdapter(adapter_spec, tokenizer_service)
        else:
            raise ValueError(f"Invalid adaptation method: {method}")

        return adapter
