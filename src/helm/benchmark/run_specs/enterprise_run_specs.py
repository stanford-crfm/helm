"""Run spec functions for HELM Enterprise scenarios."""

from typing import List
from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _get_weighted_classification_metric_specs(labels: List[str]) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.classification_metrics.ClassificationMetric",
            args={"averages": ["weighted"], "scores": ["f1", "precision", "recall"], "labels": labels},
        )
    ]


# Finance


@run_spec_function("gold_commodity_news")
def get_news_headline_spec(category: str) -> RunSpec:
    from helm.benchmark.scenarios.gold_commodity_news_scenario import GoldCommodityNewsScenario

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.gold_commodity_news_scenario.GoldCommodityNewsScenario",
        args={"category": category},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=GoldCommodityNewsScenario.get_instructions(category), input_noun="Headline", output_noun="Answer"
    )

    return RunSpec(
        name=f"gold_commodity_news:category={category}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + _get_weighted_classification_metric_specs(labels=["yes", "no"]),
        groups=["gold_commodity_news"],
    )


@run_spec_function("financial_phrasebank")
def get_financial_phrasebank_spec(agreement: int = 50) -> RunSpec:
    from helm.benchmark.scenarios.financial_phrasebank_scenario import FinancialPhrasebankScenario

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.financial_phrasebank_scenario.FinancialPhrasebankScenario",
        args={"agreement": agreement},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=FinancialPhrasebankScenario.INSTRUCTIONS,
        input_noun="Sentence",
        output_noun="Label",
        max_tokens=30,
    )

    return RunSpec(
        name=f"financial_phrasebank:agreement={agreement}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs()
        + _get_weighted_classification_metric_specs(labels=["positive", "neutral", "negative"]),
        groups=["financial_phrasebank"],
    )


@run_spec_function("conv_fin_qa_calc")
def get_conv_fin_qa_calc_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.conv_fin_qa_calc_scenario.ConvFinQACalcScenario", args={}
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Based on the table, answer the final question. Respond with the answer only, with no additional explanation.",  # noqa: E501
        input_noun=None,
        output_noun="Answer",
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.conv_fin_qa_calc_metrics.ConvFinQACalcMetric")
    ] + get_basic_metric_specs([])

    return RunSpec(
        name="conv_fin_qa_calc",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["conv_fin_qa_calc"],
    )


# Legal


@run_spec_function("legal_contract_summarization")
def get_legal_contract_summarization_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_contract_summarization_scenario.LegalContractSummarizationScenario",
        args={},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Summarize the legal document in plain English.",
        input_noun="Document",
        output_noun="Summary",
        max_tokens=100,
        stop_sequences=["\n\n"],
    )

    return RunSpec(
        name="legal_contract_summarization",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["rouge_1", "rouge_2", "rouge_l"]),
        groups=["legal_contract_summarization"],
    )


@run_spec_function("legal_opinion_sentiment_classification")
def get_legal_opinion_sentiment_classification_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_opinion_sentiment_classification_scenario.LegalOpinionSentimentClassificationScenario",  # noqa: E501
    )

    instructions = "Classify the sentences into one of the 3 sentiment categories. Possible labels: positive, neutral, negative."  # noqa: E501
    adapter_spec = get_generation_adapter_spec(
        instructions=instructions,
        output_noun="Label",
    )

    return RunSpec(
        name="legal_opinion_sentiment_classification",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs()
        + _get_weighted_classification_metric_specs(labels=["positive", "neutral", "negative"]),
        groups=["legal_opinion_sentiment_classification"],
    )


@run_spec_function("casehold")
def get_casehold_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.casehold_scenario.CaseHOLDScenario", args={})

    method = ADAPT_MULTIPLE_CHOICE_JOINT
    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="Give a letter answer among A, B, C, D, or E.",
        input_noun="Passage",
        output_noun="Answer",
        max_train_instances=2,
    )

    return RunSpec(
        name="casehold",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["casehold"],
    )


@run_spec_function("echr_judgment_classification")
def get_echr_judgment_classification_spec() -> RunSpec:
    """A different implementation (binary classification) of lex_glue_fixed:subset=ecthr_a"""
    from helm.benchmark.scenarios.echr_judgment_classification_scenario import EchrJudgeScenario

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.echr_judgment_classification_scenario.EchrJudgeScenario",
        args={"filter_max_length": 600},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=EchrJudgeScenario.PROMPT_INST_WITH_EX,
        input_noun=EchrJudgeScenario.PROMPT_INPUT,
        output_noun=EchrJudgeScenario.PROMPT_OUTPUT,
        max_tokens=1,
    )

    return RunSpec(
        name="echr_judgment_classification",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + _get_weighted_classification_metric_specs(labels=["yes", "no"]),
        groups=["echr_judgment_classification"],
    )


# Climate


@run_spec_function("sumosum")
def get_sumosum_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.sumosum_scenario.SUMOSumScenario",
        args={
            # A too-short article could be garbage.
            "test_filter_min_length": 100,
            # A too-long article doesn't fit in a prompt.
            "test_filter_max_length": 3700,
        },
    )

    instructions = "Generate the title of the following article."
    adapter_spec = get_generation_adapter_spec(
        instructions=instructions,
        output_noun="Title",
        max_train_instances=0,
        max_tokens=100,
        stop_sequences=["\n\n"],
    )

    return RunSpec(
        name="sumosum",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_metric_specs(["rouge_1", "rouge_2", "rouge_l"]),
        groups=["sumosum"],
    )


# Cyber Security


@run_spec_function("cti_to_mitre")
def get_cti_to_mitre_spec(num_options: int = 10, seed: int = 42, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cti_to_mitre_scenario.CtiToMitreScenario",
        args={
            "num_options": num_options,
            "seed": seed,
        },
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="Classify the following situation by the type of security attack. Answer with only a single letter.",  # noqa:
        input_noun="Situation",
        output_noun="Answer",
        max_train_instances=10,
    )

    return RunSpec(
        name=f"cti_to_mitre:num_options={num_options},seed={seed},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["cti_to_mitre"],
    )
