from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List

from benchmark.scenario import Instance, Reference
from common.object_spec import ObjectSpec, create_object


class Perturbation(ABC):

    # Unique name to describe perturbation. We use the name to tag instances.
    name: str

    def apply(self, id_tag: str, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        if should_perturb_references is true.
        """
        references: List[Reference] = instance.references
        if should_perturb_references:
            references = [self.perturb_reference(reference) for reference in references]

        return replace(
            instance,
            input=self.perturb(instance.input),
            tags=instance.tags + [id_tag, self.tag],
            references=references,
        )

    def perturb_reference(self, reference: Reference) -> Reference:
        """Generates a new Reference by perturbing the output and tagging the Reference."""
        return replace(reference, output=self.perturb(reference.output), tags=reference.tags + [self.tag])

    @abstractmethod
    def perturb(self, text: str) -> str:
        """How to perturb the text. """
        pass

    @property
    def tag(self) -> str:
        """Used to tag instances to indicate which perturbation has been applied."""
        return self.name


class PerturbationSpec(ObjectSpec):
    """Defines how to instantiate Perturbation."""

    pass


def create_perturbation(perturbation_spec: PerturbationSpec) -> Perturbation:
    """Creates Perturbation from PerturbationSpec."""
    return create_object(perturbation_spec)


# TODO: Get rid of this after we add the new instance fields:
#       https://github.com/stanford-crfm/benchmarking/issues/124
@dataclass
class CleanPerturbation(Perturbation):
    """Doesn't apply any perturbation, but just adds 'clean' to the list of tags."""

    CLEAN_TAG = "clean"

    name = CLEAN_TAG

    def perturb(self, text: str) -> str:
        return text


@dataclass
class ExtraSpacePerturbation(Perturbation):
    """A toy perturbation that adds additional spaces to existing spaces."""

    name = "extra_space"

    def __init__(self, num_spaces: int):
        self.num_spaces = num_spaces

    def perturb(self, text: str) -> str:
        return text.replace(" ", " " * self.num_spaces)

    @property
    def tag(self) -> str:
        return f"{self.name}|num_spaces={self.num_spaces}"
