"""Run spec functions for HELM Enterprise scenarios."""

from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_classification_metric_specs,
    get_exact_match_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


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
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs(),
        groups=["gold_commodity_news"],
    )

@run_spec_function("legal_contract_summarization")
def get_legal_contract_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_contract_summarization_scenario.LegalContractSummzarizationScenario",
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
        metric_specs=get_basic_metric_specs(["rouge_1", "rouge_2"]),
        groups=["legal_contract_summarization"],
    )