from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

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

    # The output text
    output: str

    # Extra metadata (e.g., whether it's correct/factual/toxic)
    tags: List[str]

    def __repr__(self):
        return f"Reference ({', '.join(self.tags)}): {self.output}"

    @property
    def is_correct(self) -> bool:
        return CORRECT_TAG in self.tags


@dataclass(frozen=True, eq=False)
class Instance:
    """
    An `Instance` represents one data point that we're evaluating on (e.g., one
    question in a QA task).
    Note: `eq=False` means that we hash by the identity.
    """

    # The input text
    input: str

    # References that helps us evaluate
    references: List[Reference]

    # Extra metadata (e.g., train/valid/test, demographic group, etc.)
    tags: List[str]

    @property
    def first_correct_reference(self) -> Optional[Reference]:
        """Return the first correct reference."""
        for reference in self.references:
            if reference.is_correct:
                return reference
        return None

    def info(self) -> List[str]:
        info = [f"Input: {self.input}"]
        for reference in self.references:
            info.append(str(reference))
        return info


@dataclass  # type: ignore
class Scenario(ABC):
    """
    A scenario represents a (task, data distribution).
    It is usually based on some raw dataset and is converted into a list of `Instance`s.
    Override this class.

    Note: the constructor should be lightweight, `get_instances` should do all
    the heavy lifting.
    """

    # Short unique identifier of the scenario (e.g., RealToxicityPrompts)
    name: str

    # Description of the scenario (task, data)
    description: str

    # Extra metadata (e.g., whether this is a question answering or commonsense task)
    tags: List[str]

    # To be set by the `Runner`
    output_path: str = ""

    @abstractmethod
    def get_instances(self) -> List[Instance]:
        """
        Does the main work in the `Scenario` (e.g., download datasets, convert
        it into a list of instances).
        """
        pass

    def info(self, instances: List[Instance]) -> List[str]:
        total = len(instances)
        output = [
            f"Scenario: {self.name}",
            self.description,
            f"Tags: {', '.join(self.tags)}",
            f"{total} instances",
            "",
        ]

        for i, instance in enumerate(instances):
            output.append(f"------- Instance {i + 1}/{total}: {', '.join(instance.tags)}")
            output.extend(instance.info())
            output.append("")
        return output


class ScenarioSpec(ObjectSpec):
    pass


def create_scenario(scenario_spec: ScenarioSpec) -> Scenario:
    return create_object(scenario_spec)
