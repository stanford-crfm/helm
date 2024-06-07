import json
import os
from typing import Any, Dict
from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.common.general import ensure_file_downloaded


_RECIPES_BASE_URL = "https://raw.githubusercontent.com/aiverify-foundation/moonshot-data/main/recipes"


def download_recipe(recipe_id) -> Dict[str, Any]:
    file_name = f"{recipe_id}.json"
    source_url = f"{_RECIPES_BASE_URL}/{file_name}"
    file_path = os.path.join("moonshot-data", file_name)

    ensure_file_downloaded(source_url=source_url, target_path=file_path)

    with open(file_path, "r") as f:
        return json.load(f)


@run_spec_function("moonshot")
def get_moonshot_spec(recipe: str) -> RunSpec:
    recipe_data = download_recipe(recipe)
    # https://raw.githubusercontent.com/aiverify-foundation/moonshot-data/main/recipes/
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        global_prefix="",
        global_suffix="",
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=512,
        temperature=0.0,
        stop_sequences=[],
    )
    print(recipe_data["datasets"])
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.moonshot_scenario.MoonshotScenario",
        args={"dataset_ids": recipe_data["datasets"]},
    )
    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.moonshot_metrics.MoonshotMetric", args={"metric_id": metric_id})
        for metric_id in recipe_data["metrics"]
    ]
    recipe_group = "moonshot_" + recipe.replace("-", "_")
    return RunSpec(
        name=f"moonshot:recipe={recipe}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["moonshot", recipe_group],
    )
