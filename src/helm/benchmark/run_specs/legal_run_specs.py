from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_generative_harms_metric_specs,
    get_summarization_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path

from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path


@run_spec_function("lsat_qa")
def get_lsat_qa_spec(task: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lsat_qa_scenario.LSATScenario", args={"task": task}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="The following are multiple choice questions (with answers).",
        input_noun="Passage",
        output_noun="Answer",
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"lsat_qa:task={task},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lsat_qa"],
    )


@run_spec_function("legalbench")
def get_legalbench_spec(subset: str) -> RunSpec:
    from helm.benchmark.scenarios.legalbench_scenario import (
        LegalBenchScenario,
        get_legalbench_instructions,
        get_legalbench_output_nouns,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legalbench_scenario.LegalBenchScenario", args={"subset": subset}
    )
    scenario_cache_path = get_scenario_cache_path(get_benchmark_output_path(), LegalBenchScenario.name)
    adapter_spec = get_generation_adapter_spec(
        instructions=get_legalbench_instructions(subset, scenario_cache_path),
        input_noun=None,
        output_noun=get_legalbench_output_nouns(subset, scenario_cache_path),
        max_tokens=30,  # at most ~50 characters per label,
        max_train_instances=5,  # Use 5 for all subsets
    )

    return RunSpec(
        name=f"legalbench:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["legalbench"],
    )


@run_spec_function("legal_support")
def get_legal_support_spec(method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_support_scenario.LegalSupportScenario", args={}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions="Which statement best supports the passage?",
        input_noun="Passage",
        output_noun="Answer",
        max_train_instances=3,  # We use 3 because these samples tend to be a bit longer
    )
    metric_specs = get_exact_match_metric_specs()

    return RunSpec(
        name=f"legal_support,method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["legal_support"],
    )


@run_spec_function("lextreme")
def get_lextreme_spec(subset: str) -> RunSpec:
    from helm.benchmark.scenarios.lextreme_scenario import (
        TaskType,
        get_lextreme_instructions,
        get_lextreme_max_tokens,
        get_lextreme_max_train_instances,
        get_lextreme_task_type,
    )

    task_type = get_lextreme_task_type(subset)

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lextreme_scenario.LEXTREMEScenario",
        args={"subset": subset},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=get_lextreme_instructions(subset),
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=get_lextreme_max_tokens(subset),
        max_train_instances=get_lextreme_max_train_instances(subset),  # in some subsets the input is very long
        multi_label=(task_type == TaskType.MLTC),
    )

    metric_specs = get_basic_metric_specs([])
    if task_type == TaskType.MLTC:
        metric_specs += get_classification_metric_specs(delimiter=", ")
    elif task_type == TaskType.SLTC:
        metric_specs += get_classification_metric_specs()

    return RunSpec(
        name=f"lextreme:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lextreme"],
    )


@run_spec_function("lex_glue")
def get_lex_glue_spec(subset: str) -> RunSpec:
    from helm.benchmark.scenarios.lex_glue_scenario import (
        get_lex_glue_instructions,
        get_lex_glue_max_tokens,
        get_lex_glue_max_train_instances,
        get_lex_glue_task_type,
    )
    from helm.benchmark.scenarios.lextreme_scenario import TaskType

    task_type = get_lex_glue_task_type(subset)

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.lex_glue_scenario.LexGLUEScenario",
        args={"subset": subset},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=get_lex_glue_instructions(subset),
        input_noun="Passage",
        output_noun="Answer",
        max_tokens=get_lex_glue_max_tokens(subset),
        max_train_instances=get_lex_glue_max_train_instances(subset),  # in some subsets the input is very long
        multi_label=(task_type == TaskType.MLTC),
    )

    metric_specs = get_basic_metric_specs([])
    if task_type == TaskType.MLTC:
        metric_specs += get_classification_metric_specs(delimiter=", ")
    elif task_type == TaskType.SLTC:
        metric_specs += get_classification_metric_specs()

    return RunSpec(
        name=f"lex_glue:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lex_glue"],
    )


@run_spec_function("billsum_legal_summarization")
def get_billsum_legal_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_summarization_scenario.LegalSummarizationScenario",
        args={
            "dataset_name": "BillSum",
            "sampling_min_length": 200,
            "sampling_max_length": 800,  # 2000 would be ideal, but for economic reasons set it lower
            "doc_max_length": 2048,  # 4096 would be ideal, but for economic reasons set it lower
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=None,
        max_tokens=1024,  # From Kornilova & Eidelmann, 2020 (https://arxiv.org/pdf/1910.00523.pdf)
        temperature=temperature,  # similar to other summarization tasks
    )

    return RunSpec(
        name=f"legal_summarization:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "billsum_legal_summarization", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["legal_summarization", "summarization"],
    )


@run_spec_function("multilexsum_legal_summarization")
def get_multilexsum_legal_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_summarization_scenario.LegalSummarizationScenario",
        args={
            "dataset_name": "MultiLexSum",
            "sampling_min_length": 100,
            "sampling_max_length": 400,  # 1000 would be ideal, but for economic reasons set it lower
            "doc_max_length": 1024,  # 2048 would be ideal, but for economic reasons set it lower
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=2,
        max_tokens=256,  # From Shen et al., 2022 (https://arxiv.org/pdf/2206.10883.pdf)
        temperature=temperature,  # similar to other summarization tasks
    )

    return RunSpec(
        name=f"legal_summarization:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "multilexsum_legal_summarization", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["legal_summarization", "summarization"],
    )


@run_spec_function("eurlexsum_legal_summarization")
def get_eurlexsum_legal_summarization_spec(temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.legal_summarization_scenario.LegalSummarizationScenario",
        args={
            "dataset_name": "EurLexSum",
            "sampling_min_length": 400,
            "sampling_max_length": 1600,  # 4000 would be ideal, but for economic reasons set it lower
            "doc_max_length": 2048,  # 8192 would be ideal, but for economic reasons set it lower
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=None,
        max_tokens=2048,  # From Aumiller et al., 2022 (https://arxiv.org/pdf/2210.13448.pdf)
        temperature=temperature,  # similar to other summarization tasks
    )

    return RunSpec(
        name=f"legal_summarization:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "eurlexsum_legal_summarization", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["legal_summarization", "summarization"],
    )
