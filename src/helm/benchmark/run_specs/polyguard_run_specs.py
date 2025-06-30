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




@run_spec_function("polyguard")
def get_polyguard_spec(
    language: Optional[str],
    request_type: Optional[str] = "both",
    annotator_model: Optional[str] = None, 
    annotator_model_deployment: Optional[str] = None,
    ) -> RunSpec:
    
    run_spec_name = (
        "polyguard:" +
        f"language={language}"
        f",request_type={request_type}"
        f",annotator_model=toxicityprompts/polyguard-qwen-smol"
        f",annotator_model_deployment=toxicityprompts/polyguard-qwen-smol"
    )
    
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.polyguard_scenario.PolyGuardScenario",
        args={"language": language, "request_type": request_type},
    )

    adapter_spec: AdapterSpec = get_instruct_adapter_spec(
        num_outputs=1,
        max_tokens=512,
        temperature=0.0,
    )
    
    annotator_args: Dict[str, str] = {}
    if annotator_model:
        annotator_args["model"] = annotator_model
        annotator_args["model_deployment"] = annotator_model_deployment or annotator_model
        run_spec_name = (
            "polyguard:" +
            f"language={language}"
            f",request_type={request_type}"
            f",annotator_model={annotator_args['model']}"
            f",annotator_model_deployment={annotator_args['model_deployment']}"
        )
       
        
    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.polyguard_annotator.PolyGuardAnnotator", args=annotator_args
        )
    ]
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.polyguard_metrics.PolyGuardMetric"),
    ]

    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        annotators=annotator_specs,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["polyguard"],
    )
