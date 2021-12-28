from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from object_spec import ObjectSpec, create_object

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

    @property
    def first_correct_reference(self) -> Optional[Reference]:
        """Return the first correct reference."""
        for reference in self.references:
            if CORRECT_TAG in reference.tags:
                return reference
        return None


class Scenario(ABC):
    """
    A scenario represents a (task, data distribution).
    It is usually based on some raw dataset and is converted into a list of `Instance`s.
    Override this class.
    """

    name: str  # Short unique identifier of the scenario (e.g., RealToxicityPrompts)
    description: str  # Description of the scenario (task, data)
    tags: List[str]  # Extra metadata (e.g., whether this is a question answering or commonsense task)

    @abstractmethod
    def get_instances(self) -> List[Instance]:
        """
        Download any necessary datasets.
        Load the data and convert it into a list of instances.
        """
        pass


class ScenarioSpec(ObjectSpec):
    pass


def create_scenario(scenario_spec: ScenarioSpec) -> Scenario:
    return create_object(scenario_spec)
