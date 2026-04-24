from typing import List

from helm.benchmark.scenarios.scenario import (
    Output,
    Reference,
    Scenario,
    Instance,
    Input,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALID_SPLIT,
)
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from unitxt import load_dataset
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["unitxt"])


class UnitxtScenario(Scenario):
    """Integration with Unitxt: https://www.unitxt.ai/"""

    name = "unitxt"
    description = "Unitxt Scenarios"
    tags = ["unitxt"]

    UNITXT_SPLIT_NAME_TO_HELM_SPLIT_NAME = {
        "train": TRAIN_SPLIT,
        "test": TEST_SPLIT,
        "validation": VALID_SPLIT,
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def get_instances(self, output_path: str) -> List[Instance]:
        if len(self.kwargs) == 1 and "recipe" in self.kwargs:
            unitxt_dataset = load_dataset(self.kwargs["recipe"])
        else:
            unitxt_dataset = load_dataset(**self.kwargs)

        if isinstance(unitxt_dataset, dict):
            dataset = unitxt_dataset
        else:
            if "split" in self.kwargs:
                dataset = {self.kwargs["split"]: unitxt_dataset}
            else:
                raise Exception(
                    "Expected Unitxt load_dataset() to return a dict because `split` was not specified, but instead it returned a bare dataset"
                )

        instances: List[Instance] = []

        for unitxt_split_name, helm_split_name in UnitxtScenario.UNITXT_SPLIT_NAME_TO_HELM_SPLIT_NAME.items():
            dataset_split = dataset.get(unitxt_split_name)
            if dataset_split is None:
                continue
            for index, row in enumerate(dataset_split):
                references = [
                    Reference(
                        output=Output(text=reference_text),
                        tags=[CORRECT_TAG],
                    )
                    for reference_text in row["references"]
                ]
                instance = Instance(
                    id=f"{unitxt_split_name}{index}",
                    input=Input(text=row["source"]),
                    references=references,
                    split=helm_split_name,
                )
                instances.append(instance)
        return instances
