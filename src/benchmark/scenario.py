import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from common.object_spec import ObjectSpec, create_object

# Tags for instances
TRAIN_TAG = "train"
VALID_TAG = "valid"
TEST_TAG = "test"

# Tags for references
CORRECT_TAG = "correct"


@dataclass(frozen=True)
class Reference:
    """
    A `Reference` specifies a possible output and how good/bad it is.  This
    could be used to represent multiple reference outputs which are all
    acceptable (e.g., in machine translation) or alternatives (e.g., in a
    multiple-choice exam).
    """

    output: str  # The output text
    tags: List[str]  # Extra metadata (e.g., whether it's correct/factual/toxic)

    def __str__(self):
        return f"Reference ({', '.join(self.tags)}): {self.output}"

    @property
    def is_correct(self) -> bool:
        return CORRECT_TAG in self.tags

    def to_dict(self):
        return {"output": self.output, "tags": self.tags}


@dataclass(frozen=True, eq=False)
class Instance:
    """
    An `Instance` represents one data point that we're evaluating on (e.g., one
    question in a QA task).
    Note: `eq=False` means that we hash by the identity.
    """

    input: str  # The input text
    references: List[Reference]  # References that helps us evaluate
    tags: List[str]  # Extra metadata (e.g., train/valid/test, demographic group, etc.)

    def __str__(self):
        return f"Input: {self.input}\n" + "\n".join(str(reference) for reference in self.references)

    @property
    def first_correct_reference(self) -> Optional[Reference]:
        """Return the first correct reference."""
        for reference in self.references:
            if reference.is_correct:
                return reference
        return None

    def to_dict(self) -> Dict:
        return {
            "input": self.input,
            "tags": self.tags,
            "references": [reference.to_dict() for reference in self.references],
        }


class Scenario(ABC):
    """
    A scenario represents a (task, data distribution).
    It is usually based on some raw dataset and is converted into a list of `Instance`s.
    Override this class.
    """

    name: str  # Short unique identifier of the scenario (e.g., RealToxicityPrompts)
    description: str  # Description of the scenario (task, data)
    tags: List[str]  # Extra metadata (e.g., whether this is a question answering or commonsense task)

    def __str__(self) -> str:
        """
        Converts the Scenario into a string for pretty printing.
        """
        instances: List[Instance] = self.get_instances()
        total = len(instances)

        output: str = f"Scenario: {self.name}\n{self.description}\nTags: {', '.join(self.tags)}\n{total} instances"
        for i, instance in enumerate(instances):
            output += f"\n\n------- Instance {i+1}/{total}: {', '.join(instance.tags)}\n{instance}"
        return output

    @abstractmethod
    def get_instances(self) -> List[Instance]:
        """
        Download any necessary datasets.
        Load the data and convert it into a list of instances.
        """
        pass

    def to_dict(self) -> Dict:
        instances: List[Instance] = self.get_instances()
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "instances": [instance.to_dict() for instance in instances],
        }

    def to_json(self, pretty=False) -> str:
        """
        Converts `Scenario` into JSON string.
        """
        return json.dumps(self.to_dict(), indent=4) if pretty else json.dumps(self.to_dict())


class ScenarioSpec(ObjectSpec):
    pass


def create_scenario(scenario_spec: ScenarioSpec) -> Scenario:
    return create_object(scenario_spec)
