# mypy: check_untyped_defs = False
from typing import List

from helm.benchmark.augmentations.data_augmenter import DataAugmenterSpec
from helm.benchmark.augmentations.perturbation import PerturbationSpec
from helm.benchmark.data_preprocessor import DataPreprocessor
from helm.benchmark.run_specs.simple_run_specs import get_simple1_spec
from helm.benchmark.scenarios.scenario import create_scenario, Instance, Scenario, with_instance_ids


def test_data_preprocessor():
    # Test that each Instance is given a unique ID and is preserved through data augmentation
    data_preprocessor = DataPreprocessor(DataAugmenterSpec())
    scenario: Scenario = create_scenario(get_simple1_spec().scenario_spec)
    instances = with_instance_ids(scenario.get_instances(output_path=""))
    instances: List[Instance] = data_preprocessor.preprocess(instances)
    for i, instance in enumerate(instances):
        assert instance.id == f"id{i}"


def test_data_preprocessor_with_data_augmentation():
    data_augmenter_spec = DataAugmenterSpec(
        perturbation_specs=[
            PerturbationSpec(
                class_name="helm.benchmark.augmentations.extra_space_perturbation.ExtraSpacePerturbation",
                args={"num_spaces": 5},
            )
        ],
        should_augment_train_instances=False,
        should_include_original_train=False,
        should_augment_eval_instances=True,
        should_include_original_eval=True,
    )
    data_preprocessor = DataPreprocessor(data_augmenter_spec)
    scenario: Scenario = create_scenario(get_simple1_spec().scenario_spec)
    instances = with_instance_ids(scenario.get_instances(output_path=""))
    instances: List[Instance] = data_preprocessor.preprocess(instances)
    assert len(instances) == 10 + 10 + 10  # original train + original eval + perturbed eval

    # After the data preprocessing, check that the data augmentation has been applied
    # by verifying that the instances with the perturbation tag are perturbed
    for instance in instances:
        if instance.perturbation and instance.perturbation.name == "extra_space":
            assert " " * 5 in instance.input.text
        else:
            assert " " * 5 not in instance.input.text
