from typing import Optional, Dict

from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import (
    get_instruct_adapter_spec
)
from helm.benchmark.adaptation.adapter_spec import (
    AdapterSpec,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec




@run_spec_function("eclektic")
def get_eclektic_spec(
    annotator_model: Optional[str] = "qwen/qwen2.5-0.5b-instruct", 
    annotator_model_deployment: Optional[str] = "huggingface/qwen2.5-0.5b-instruct",
    ) -> RunSpec:
    
    annotator_args: Dict[str, str] = {}
    annotator_args["model"] = annotator_model
    annotator_args["model_deployment"] = annotator_model_deployment or annotator_model
    run_spec_name = (
        "eclektic:" +
        f"annotator_model={annotator_args['model']}"
        f",annotator_model_deployment={annotator_args['model_deployment']}"
    )
    
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.eclektic_scenario.EclekticScenario",
    )

    adapter_spec: AdapterSpec = get_instruct_adapter_spec(
        num_outputs=1,
        max_tokens=50,
        temperature=0.0,
    )
    
    
    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.eclektic_annotator.EclekticAnnotator", args=annotator_args
        )
    ]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.eclektic_metrics.EclekticMetric"),
    ]

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        annotators=annotator_specs,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["eclektic"],
    )
