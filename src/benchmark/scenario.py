from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from common.object_spec import ObjectSpec, create_object
from common.general import format_text, format_split, format_tags, indent_lines
from benchmark.augmentations.perturbation_description import PerturbationDescription

# Data splits
TRAIN_SPLIT: str = "train"
VALID_SPLIT: str = "valid"
TEST_SPLIT: str = "test"
EVAL_SPLITS: List[str] = [VALID_SPLIT, TEST_SPLIT]

# We mainly care about having enough test examples to ensure statistical significance -- the remaining N-1000 instances become training examples.
DEFAULT_TEST_SIZE: int = 1000

# Tags for references
CORRECT_TAG: str = "correct"


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

    @property
    def is_correct(self) -> bool:
        return CORRECT_TAG in self.tags

    def render_lines(self) -> List[str]:
        return [f"reference {format_tags(self.tags)}: {format_text(self.output)}"]


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

    # Split (e.g., train, valid, test)
    split: Optional[str] = None

    # Sub split (e.g. toxic, non-toxic)
    sub_split: Optional[str] = None

    # Used to group Instances that were created from a particular Instance through data augmentation
    id: Optional[str] = None

    # Description of the Perturbation that was applied when creating this Instance
    perturbation: Optional[PerturbationDescription] = None

    @property
    def first_correct_reference(self) -> Optional[Reference]:
        """Return the first correct reference."""
        for reference in self.references:
            if reference.is_correct:
                return reference
        return None

    def render_lines(self) -> List[str]:
        info = [f"input: {format_text(self.input)}"]
        if self.sub_split:
            info.append(f"sub_split: {format_text(self.sub_split)}")
        if self.id:
            info.append(f"id: {format_text(self.id)}")
        if self.perturbation:
            info.append(f"perturbation: {self.perturbation}")

        for reference in self.references:
            info.extend(reference.render_lines())
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

    # To be set by the `Runner` (for caching data)
    output_path: str

    @abstractmethod
    def get_instances(self) -> List[Instance]:
        """
        Does the main work in the `Scenario` (e.g., download datasets, convert
        it into a list of instances).
        """
        pass

    def render_lines(self, instances: List[Instance]) -> List[str]:
        total = len(instances)
        output = [
            f"name: {self.name}",
            f"description: {self.description}",
            f"tags: {format_tags(self.tags)}",
            "",
        ]

        for i, instance in enumerate(instances):
            output.append(f"instance {i} ({total} total) {format_split(instance.split)} {{")
            output.extend(indent_lines(instance.render_lines()))
            output.append("}")
        return output


class ScenarioSpec(ObjectSpec):
    pass


def create_scenario(scenario_spec: ScenarioSpec) -> Scenario:
    return create_object(scenario_spec)
