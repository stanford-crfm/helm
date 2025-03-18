from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import PurePath
import inspect

from helm.common.media_object import MultimediaObject
from helm.common.object_spec import ObjectSpec, create_object
from helm.common.general import ensure_directory_exists, format_text, format_split, format_tags, indent_lines
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription

""" Data splits """
TRAIN_SPLIT: str = "train"
VALID_SPLIT: str = "valid"
TEST_SPLIT: str = "test"
EVAL_SPLITS: List[str] = [VALID_SPLIT, TEST_SPLIT]
ALL_SPLITS: List[str] = [TRAIN_SPLIT] + EVAL_SPLITS

""" Number of examples """
# We mainly care about having enough test examples to ensure statistical significance;
# the remaining N-1000 instances become training examples.
DEFAULT_TEST_SIZE: int = 1000

""" Reference tags """
CORRECT_TAG: str = "correct"

""" Asset tags (used for compiled outputs such as image2struct)"""
ASSET_NAME_TAG: str = "asset_name"
ASSET_PATH_TAG: str = "asset_path"

# Reference tag functions for ranking scenarios.
# @TODO: (For future) Should there be a base RankingScenario class?


def make_relevance_tag(relevance: int) -> str:
    """Make a relevance tag.

    Relevance value is an integer bigger than or equal to 0.
    """
    return f"relevance={relevance}"


def make_rank_tag(rank: int) -> str:
    """Make a rank tag.

    Rank value is an integer bigger than or equal to 1.
    """
    return f"rank={rank}"


def unpack_tag(tag: str) -> Tuple[str, str]:
    """Unpack the value from the tag."""
    key, value = tag.split("=")
    return key, value


@dataclass(frozen=True)
class Input:
    """
    The input of an `Instance`.
    """

    text: str = ""
    """The text of the input (e.g, passage to summarize, text for sentiment analysis, etc.)"""

    multimedia_content: Optional[MultimediaObject] = None
    """A single input can consists of multimodal content interleaved (e.g., text, image, text, ...)."""

    messages: Optional[List[Dict[str, str]]] = None
    """Used for chat models.
    If messages is specified for a chat model, the prompt is ignored.
    Otherwise, the client should convert the prompt into a message."""


@dataclass(frozen=True)
class PassageQuestionInput(Input):
    """
    Passage-question pair used for question answering Scenarios.
    """

    def __init__(
        self,
        passage: str,
        question: str,
        passage_prefix: str = "",
        question_prefix: str = "Question: ",
        separator: str = "\n",
    ):
        super().__init__(f"{passage_prefix}{passage}{separator}{question_prefix}{question}")


@dataclass(frozen=True)
class Output:
    """
    The output of a `Reference`.
    """

    text: str = ""
    """The text of the output."""

    multimedia_content: Optional[MultimediaObject] = None
    """The output can be multimodal content interleaved (e.g., text, image, text, ...)."""


@dataclass(frozen=True)
class Reference:
    """
    A `Reference` specifies a possible output and how good/bad it is.  This
    could be used to represent multiple reference outputs which are all
    acceptable (e.g., in machine translation) or alternatives (e.g., in a
    multiple-choice exam).
    """

    output: Output
    """The output"""

    tags: List[str]
    """Extra metadata (e.g., whether it's correct/factual/toxic)"""

    @property
    def is_correct(self) -> bool:
        return CORRECT_TAG in self.tags

    def render_lines(self) -> List[str]:
        return [f"reference {format_tags(self.tags)}: {format_text(self.output.text)}"]


@dataclass(frozen=True, eq=False)
class Instance:
    """
    An `Instance` represents one data point that we're evaluating on (e.g., one
    question in a QA task).
    Note: `eq=False` means that we hash by the identity.
    """

    input: Input
    """The input"""

    references: List[Reference]
    """References that helps us evaluate"""

    split: Optional[str] = None
    """Split (e.g., train, valid, test)"""

    sub_split: Optional[str] = None
    """Sub split (e.g. toxic, non-toxic)"""

    id: Optional[str] = None
    """Used to group Instances that were created from a particular Instance through data augmentation"""

    perturbation: Optional[PerturbationDescription] = None
    """Description of the Perturbation that was applied when creating this Instance"""

    contrast_inputs: Optional[List[Input]] = None
    """Perturbed input as defined by contrast sets (if available)"""

    contrast_references: Optional[List[List[Reference]]] = None
    """References for the perturbed input above (if available)"""

    extra_data: Optional[Dict[str, Any]] = None
    """Extra data required by the scenario e.g. chain-of-thought annotations"""

    @property
    def first_correct_reference(self) -> Optional[Reference]:
        """Return the first correct reference."""
        for reference in self.references:
            if reference.is_correct:
                return reference
        return None

    @property
    def all_correct_references(self) -> List[Reference]:
        """Return all correct references."""
        return [reference for reference in self.references if reference.is_correct]

    def render_lines(self) -> List[str]:
        info = [f"input: {format_text(self.input.text)}"]
        if self.sub_split:
            info.append(f"sub_split: {format_text(self.sub_split)}")
        if self.id:
            info.append(f"id: {format_text(self.id)}")
        if self.perturbation:
            info.append(f"perturbation: {self.perturbation}")

        for reference in self.references:
            info.extend(reference.render_lines())

        return info


# TODO(#1212): Scenario should not be a dataclass.
@dataclass
class Scenario(ABC):
    """
    A scenario represents a (task, data distribution).
    It is usually based on some raw dataset and is converted into a list of `Instance`s.
    Override this class.

    Note: the constructor should be lightweight, `get_instances` should do all
    the heavy lifting.
    """

    # Set by the Scenario subclass.
    name: str = field(init=False)
    """Short unique identifier of the scenario"""

    # Set by the Scenario subclass.
    description: str = field(init=False)
    """Description of the scenario (task, data)"""

    # Set by the Scenario subclass.
    tags: List[str] = field(init=False)
    """Extra metadata (e.g., whether this is a question answering or commonsense task)"""

    definition_path: str = field(init=False)
    """Where the scenario subclass for `self` is defined."""

    def __post_init__(self) -> None:
        parts = list(PurePath(inspect.getfile(type(self))).parts)
        path = parts.pop()
        parts.reverse()
        for part in parts:
            path = part + "/" + path
            if part == "helm":
                break
        self.definition_path = "https://github.com/stanford-crfm/helm/blob/main/src/" + path

    @abstractmethod
    def get_instances(self, output_path: str) -> List[Instance]:
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
            output.append(f"instance {i} ({total} total) {format_split(str(instance.split))} {{")
            output.extend(indent_lines(instance.render_lines()))
            output.append("}")
        return output


def with_instance_ids(instances: List[Instance]) -> List[Instance]:
    """Return the instances with an ID.  Note: order of instances matters."""
    return [replace(instance, id=f"id{i}") for i, instance in enumerate(instances)]


class ScenarioSpec(ObjectSpec):
    pass


def create_scenario(scenario_spec: ScenarioSpec) -> Scenario:
    """Construct the scenario and set some fields."""
    return create_object(scenario_spec)


def get_scenario_cache_path(benchmark_output_path: str, scenario_name: str):
    """Return a directory under benchmark_output_path in which Scenario can cache temporary data."""
    scenarios_path: str = os.path.join(benchmark_output_path, "scenarios", scenario_name)
    ensure_directory_exists(scenarios_path)
    return scenarios_path
