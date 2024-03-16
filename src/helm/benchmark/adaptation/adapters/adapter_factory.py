from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_GENERATION_MULTIMODAL,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_RANKING_BINARY,
    AdapterSpec,
)
from helm.benchmark.adaptation.adapters.adapter import Adapter
from helm.benchmark.adaptation.adapters.binary_ranking_adapter import BinaryRankingAdapter
from helm.benchmark.adaptation.adapters.generation_adapter import GenerationAdapter
from helm.benchmark.adaptation.adapters.language_modeling_adapter import LanguageModelingAdapter
from helm.benchmark.adaptation.adapters.multimodal.generation_multimodal_adapter import GenerationMultimodalAdapter
from helm.benchmark.adaptation.adapters.multimodal.multiple_choice_joint_multimodal_adapter import (
    MultipleChoiceJointMultimodalAdapter,
)
from helm.benchmark.adaptation.adapters.multiple_choice_calibrated_adapter import MultipleChoiceCalibratedAdapter
from helm.benchmark.adaptation.adapters.multiple_choice_joint_adapter import MultipleChoiceJointAdapter
from helm.benchmark.adaptation.adapters.multiple_choice_separate_adapter import MultipleChoiceSeparateAdapter
from helm.benchmark.window_services.tokenizer_service import TokenizerService


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
        elif method == ADAPT_GENERATION_MULTIMODAL:
            adapter = GenerationMultimodalAdapter(adapter_spec, tokenizer_service)
        elif method == ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL:
            adapter = MultipleChoiceJointMultimodalAdapter(adapter_spec, tokenizer_service)
        else:
            raise ValueError(f"Invalid adaptation method: {method}")

        return adapter
