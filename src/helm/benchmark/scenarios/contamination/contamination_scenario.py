from typing import List
from helm.benchmark.scenarios.contamination.contamination_strategy import get_contamination_strategy
from helm.benchmark.scenarios.scenario import Scenario, Instance, create_scenario, get_scenario_cache_path
from helm.benchmark.run_spec import get_run_spec_function
from helm.common.hierarchical_logger import hlog


class ContaminationScenario(Scenario):
    """
    Meta-scenario that applies contamination strategies to existing datasets.
    Loads the original instances and applies masking based on the chosen strategy.
    """

    name = "contamination"
    description = "Meta-scenario that runs contamination analysis over another dataset."
    tags = ["contamination", "analysis", "mask"]

    def __init__(self, dataset: str, strategy: str, language: str):
        super().__init__()
        self.dataset = dataset
        self.strategy = strategy
        self.language = language

    def get_instances(self, output_path: str) -> List[Instance]:
        # Parse the dataset name and parameters
        function_name, params = self._parse_dataset_string(self.dataset)

        # Retrieve and execute the corresponding run spec function
        original_run_spec = self._get_original_run_spec(function_name, params)

        # Create the original scenario and obtain its instances
        original_instances = self._load_original_instances(original_run_spec, output_path)

        # Apply the selected contamination strategy
        contaminated_instances = self._apply_contamination_strategy(original_instances, original_run_spec)

        return contaminated_instances

    def _parse_dataset_string(self, dataset: str) -> tuple:
        """Parses the dataset string to extract the function name and its parameters."""
        if ":" in dataset:
            function_name = dataset.split(":", 1)[0]
            params_str = dataset.split(":", 1)[1]

            params = {}
            for param in params_str.split(","):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
        else:
            function_name = dataset
            params = {}

        return function_name, params

    def _get_original_run_spec(self, function_name: str, params: dict):
        """Retrieves and executes the dataset's run spec function."""
        run_spec_function = get_run_spec_function(function_name)

        if run_spec_function is None:
            error_msg = f"Unknown dataset function '{function_name}'. No run_spec_function found."
            hlog(f"[ContaminationScenario] ERROR: {error_msg}")
            raise ValueError(error_msg)

        # Call the run spec function with or without parameters
        if params:
            return run_spec_function(**params)
        else:
            return run_spec_function()

    def _load_original_instances(self, original_run_spec, output_path: str) -> List[Instance]:
        """Loads the instances from the original scenario."""
        original_scenario = create_scenario(original_run_spec.scenario_spec)

        # Retrieve cache path and instances
        scenario_cache_path = get_scenario_cache_path(output_path, original_scenario.name)
        original_instances = original_scenario.get_instances(scenario_cache_path)

        return original_instances

    def _apply_contamination_strategy(self, instances: List[Instance], original_run_spec) -> List[Instance]:
        """
        Applies the contamination strategy to the original instances.
        Uses existing strategy classes in a modular way.
        """

        # Retrieve the contamination strategy
        strategy = get_contamination_strategy(self.strategy)

        if strategy is None:
            error_msg = f"Unknown contamination strategy '{self.strategy}'"
            hlog(f"[ContaminationScenario] ERROR: {error_msg}")
            raise ValueError(error_msg)

        # Pass the language configuration to the strategy
        strategy.language = self.language

        # Apply the transformation to instances
        contaminated_instances = strategy.transform_instances(instances)

        return contaminated_instances
