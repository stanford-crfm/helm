import yaml
import json
import re

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from abc import ABC

from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_summarization_metric_specs,
)
from helm.common.gpu_utils import get_torch_device_name


SUMMARIZATION_METRICS = {
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "BERTScore-P",
    "BERTScore-R",
    "BERTScore-F",
}


@dataclass(frozen=True)
class MetricConfig(ABC):
    """Base class for all metric configurations"""

    name: str


@dataclass(frozen=True)
class SimpleMetricConfig(MetricConfig):
    """Configuration for simple string-based metrics like 'exact_match'"""

    pass


@dataclass(frozen=True)
class JuryMetricConfig(MetricConfig):
    """Configuration for jury-based metrics with multiple judges"""

    prompt_file: str
    judges: List[AnnotatorModelInfo]


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    A benchmark configuration is an immutable data structure that holds
    the configuration for a specific benchmark, including prompt, dataset and metric
    """

    name: str
    """Name of the benchmark"""

    description: str
    """Description of the benchmark"""

    prompt_file: str
    """Path to the prompt file. This prompt will be used for all instances of the benchmark."""

    dataset_file: str
    """Path to the dataset file. This dataset will be used to populate the context in the prompt."""

    main_metric: Union[SimpleMetricConfig, JuryMetricConfig]
    """The main metric for the benchmark"""

    metrics: List[Union[SimpleMetricConfig, JuryMetricConfig]]
    """List of structured metric configurations for the benchmark"""

    max_tokens: int = 1024
    """Maximum number of tokens to generate in the response"""

    def get_metric_specs(self) -> List[MetricSpec]:
        """Get the metric specifications for the benchmark"""
        metric_specs: List[MetricSpec] = []
        summarization = False
        for metric in self.metrics:
            if metric.name == "exact_match":
                metric_specs.extend(get_exact_match_metric_specs())

            elif metric.name == "jury_score":
                if not isinstance(metric, JuryMetricConfig):
                    raise AssertionError("Metric 'jury_score' must be a JuryMetricConfig")
                annotator_models = {judge.model_deployment: judge for judge in metric.judges}
                metric_specs.append(
                    MetricSpec(
                        class_name="helm.benchmark.metrics.llm_jury_metrics.LLMJuryMetric",
                        args={
                            "metric_name": "jury_score",
                            "scenario_name": self.name,
                            "annotator_models": annotator_models,
                        },
                    )
                )
                if len(self.metrics) == 1:
                    metric_specs.extend(get_basic_metric_specs([]))

            elif metric.name in SUMMARIZATION_METRICS:
                if not summarization:
                    summarization = True
                    metric_args = {
                        "task": self.name,
                        "device": get_torch_device_name(),
                        "bertscore_model": "distilbert-base-uncased",
                        "rescale_with_baseline": False,
                    }
                    metric_specs.extend(get_summarization_metric_specs(metric_args))
            else:
                raise ValueError(f"Unknown metric name: {metric.name}")
        return metric_specs

    def _get_annotation_criteria(self, prompt_template: str) -> Dict[str, List[str]]:
        criteria_tag = re.compile(r"<rubric_criteria>\s*(\{.*?\})\s*</rubric_criteria>", re.DOTALL)
        m = criteria_tag.search(prompt_template)
        if not m:
            raise ValueError("No <rubric_criteria>{...}</rubric_criteria> block found in prompt_template.")
        raw = json.loads(m.group(1))
        # normalize to Dict[str, Set[str]]
        return {k: list(v) for k, v in raw.items()}

    def get_annotator_specs(self) -> List[AnnotatorSpec]:
        """Convert jury metrics to AnnotatorSpec objects"""
        annotator_specs = []
        # return annotator_specs
        for metric in self.metrics:
            if isinstance(metric, JuryMetricConfig):
                with open(metric.prompt_file, "r") as f:
                    prompt_template = f.read()
                annotator_models = {judge.model_deployment: judge for judge in metric.judges}
                annotator_criteria = self._get_annotation_criteria(prompt_template)
                # Create a generic annotator spec - you may need to customize the class_name
                # based on your specific use case
                annotator_specs.append(
                    AnnotatorSpec(
                        class_name="helm.benchmark.annotation.model_as_judge.LLMAsJuryAnnotator",
                        args={
                            "name": self.name,
                            "prompt_template": prompt_template,
                            "annotation_criteria": annotator_criteria,
                            "annotator_models": annotator_models,
                        },
                    )
                )

        return annotator_specs


def _convert_metrics(raw_metrics: List[Dict[str, Any]]) -> List[MetricConfig]:
    """
    Convert raw metrics from YAML into structured MetricConfig objects.
    """
    converted_metrics: List[MetricConfig] = []

    for metric in raw_metrics:
        if not isinstance(metric, dict) or "name" not in metric:
            raise ValueError(
                f"Invalid metric format: {metric}. Each metric must be a dict with at least a 'name' field."
            )

        metric_name = metric["name"]

        if metric_name == "jury_score":
            if "prompt_file" not in metric or "judges" not in metric:
                raise ValueError(f"jury_score metric requires 'prompt_file' and 'judges': {metric}")

            judges = [
                AnnotatorModelInfo(
                    model_name=j["model_name"],
                    model_deployment=j["name"],
                )
                for j in metric["judges"]
            ]

            converted_metrics.append(
                JuryMetricConfig(name=metric_name, prompt_file=metric["prompt_file"], judges=judges)
            )
        else:
            converted_metrics.append(SimpleMetricConfig(name=metric_name))

    return converted_metrics


def _structure_benchmark_config(data: Dict[str, Any], cls) -> BenchmarkConfig:
    """
    Custom structure function for BenchmarkConfig that handles metrics conversion
    """
    if "metrics" in data:
        data = data.copy()  # Don't modify the original
        raw_metrics = data["metrics"]
        data["metrics"] = _convert_metrics(raw_metrics)
        data["main_metric"] = data["metrics"][0]
    else:
        raise ValueError("No metrics specified.")

    return BenchmarkConfig(
        name=data["name"],
        description=data["description"],
        prompt_file=data["prompt_file"],
        dataset_file=data["dataset_file"],
        main_metric=data["main_metric"],
        metrics=data["metrics"],
        max_tokens=data.get("max_tokens", 1024),
    )


def get_benchmark_config_from_path(path: str) -> BenchmarkConfig:
    """Load and parse benchmark configuration from YAML file"""
    with open(path) as f:
        config = yaml.safe_load(f)

    benchmark_config = _structure_benchmark_config(config, BenchmarkConfig)
    return benchmark_config
