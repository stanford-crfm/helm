import cattrs
import yaml
import json
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set
from abc import ABC

from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_language_modeling_metric_specs,
    get_classification_metric_specs,
    get_summarization_metric_specs,
)
from helm.common.gpu_utils import get_torch_device_name


@dataclass(frozen=True)
class MetricConfig(ABC):
    """Base class for all metric configurations"""

    name: str


@dataclass(frozen=True)
class SimpleMetricConfig(MetricConfig):
    """Configuration for simple string-based metrics like 'exact_match'"""

    main: bool = False


@dataclass(frozen=True)
class JuryMetricConfig(MetricConfig):
    """Configuration for jury-based metrics with multiple judges"""

    prompt_file: str
    judges: List[AnnotatorModelInfo]
    main: bool = False


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

    metrics: List[MetricConfig]
    """List of structured metric configurations for the benchmark"""

    max_tokens: int = 1024
    """Maximum number of tokens to generate in the response"""

    # Private field to store the main metric, populated after initialization
    _main_metric: Optional[MetricConfig] = field(init=False, default=None)

    @property
    def main_metric(self) -> Optional[MetricConfig]:
        """Get the main metric for this benchmark"""
        return self._main_metric

    @property
    def main_metric_name(self) -> Optional[str]:
        """Get the name of the main metric"""
        if self._main_metric is None:
            return None
        else:
            return self._main_metric.name

    def __post_init__(self):
        """Set the main metric after initialization"""
        main_metrics = [m for m in self.metrics if getattr(m, "main", False)]

        if len(main_metrics) > 1:
            raise ValueError(f"Multiple metrics marked as main: {[type(m).__name__ for m in main_metrics]}")
        elif len(main_metrics) == 1:
            object.__setattr__(self, "_main_metric", main_metrics[0])
        else:
            # No metric explicitly marked as main, use the first one as default
            object.__setattr__(self, "_main_metric", self.metrics[0] if self.metrics else None)

    def get_metric_specs(self) -> List[MetricSpec]:
        """Get the metric specifications for the benchmark"""
        metric_specs: List[MetricSpec] = []
        for metric in self.metrics:
            if metric.name == "exact_match":
                metric_specs.extend(get_exact_match_metric_specs())
            elif metric.name == "jury_score":
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
            elif metric.name == "rouge_1":
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

    def _get_annotation_criteria(self, prompt_template: str) -> Dict[str, Set[str]]:
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
                        class_name=f"helm.benchmark.annotation.model_as_judge.LLMAsJuryAnnotator",
                        args={
                            "name": f"{self.name}",
                            "prompt_template": prompt_template,
                            "annotation_criteria": annotator_criteria,
                            "annotator_models": annotator_models,
                        },
                    )
                )

        return annotator_specs


def _convert_metrics(raw_metrics: List[Union[str, Dict[str, Any]]]) -> List[MetricConfig]:
    """
    Convert raw metrics from YAML into structured MetricConfig objects
    """
    converted_metrics = []
    for i, metric in enumerate(raw_metrics):
        if isinstance(metric, str):
            # Simple string metric - cannot be marked as main via string notation
            converted_metrics.append(SimpleMetricConfig(name=metric, main=False))
        elif isinstance(metric, dict):
            # Complex metric - check the type
            if "jury_score" in metric:
                jury_config = metric["jury_score"]
                judges = []
                for judge in jury_config["judges"]:
                    # Map from YAML structure to AnnotatorModelInfo
                    # Assuming "name" in YAML maps to model_deployment
                    judges.append(AnnotatorModelInfo(model_name=judge["model_name"], model_deployment=judge["name"]))

                is_main = jury_config.get("main", False)

                converted_metrics.append(
                    JuryMetricConfig(
                        name="jury_score", prompt_file=jury_config["prompt_file"], judges=judges, main=is_main
                    )
                )
            elif isinstance(metric, dict) and len(metric) == 1:
                # Handle metrics specified as {"metric_name": {"main": true, ...}}
                metric_name, metric_config = next(iter(metric.items()))
                if isinstance(metric_config, dict):
                    is_main = metric_config.get("main", False)
                    converted_metrics.append(SimpleMetricConfig(name=metric_name, main=is_main))
                else:
                    # Fallback for other dict structures
                    converted_metrics.append(SimpleMetricConfig(name=metric_name, main=False))
            else:
                # Handle other complex metric types here as needed
                raise ValueError(f"Unknown metric type: {metric}")
        else:
            raise ValueError(f"Invalid metric format: {metric}")

    return converted_metrics


def _structure_benchmark_config(data: Dict[str, Any], cls) -> BenchmarkConfig:
    """
    Custom structure function for BenchmarkConfig that handles metrics conversion
    """
    if "metrics" in data:
        data = data.copy()  # Don't modify the original
        raw_metrics = data["metrics"]
        data["metrics"] = _convert_metrics(raw_metrics)

    return BenchmarkConfig(
        name=data["name"],
        description=data["description"],
        prompt_file=data["prompt_file"],
        dataset_file=data["dataset_file"],
        metrics=data["metrics"],
        max_tokens=data.get("max_tokens", 1024),
    )


def get_benchmark_config_from_path(path: str) -> BenchmarkConfig:
    """Load and parse benchmark configuration from YAML file"""
    with open(path) as f:
        config = yaml.safe_load(f)

    benchmark_config = _structure_benchmark_config(config, BenchmarkConfig)
    return benchmark_config


# Example usage
if __name__ == "__main__":
    config_path = "/Users/s0400266/Downloads/helm/debug/benchtest.yaml"
    benchmark_config = get_benchmark_config_from_path(config_path)

    print(f"Benchmark: {benchmark_config.name}")
    print(f"Description: {benchmark_config.description}")
    print(f"Max tokens: {benchmark_config.max_tokens}")

    # Work with metrics
    simple_metrics = get_simple_metrics(benchmark_config)
    jury_metrics = get_jury_metrics(benchmark_config)

    print(f"\nSimple metrics: {[m.name for m in simple_metrics]}")
    print(f"Jury metrics: {len(jury_metrics)}")

    for jury_metric in jury_metrics:
        print(f"  Jury prompt file: {jury_metric.prompt_file}")
        print(f"  Judges: {len(jury_metric.judges)}")
        for judge in jury_metric.judges:
            print(f"    - {judge.model_deployment} ({judge.model_name})")

    # breakpoint()  # Uncomment if you want to debug
