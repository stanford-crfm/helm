from abc import ABC, abstractmethod
from dataclasses import replace
from random import Random
from typing import Sequence


from .perturbation_description import PerturbationDescription
from benchmark.scenarios.scenario import Instance, Reference
from common.object_spec import ObjectSpec, create_object


class Perturbation(ABC):

    # Unique name to describe perturbation
    name: str

    # Whether to perturb references
    should_perturb_references: bool = False

    @property
    def description(self) -> PerturbationDescription:
        """Description of the perturbation."""
        return PerturbationDescription(name=self.name)

    def apply(self, instance: Instance) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        if should_perturb_references is true. Initializes a random number generator based on instance_id that gets
        passed to perturb and perturb_references.
        """
        assert instance.id is not None
        rng: Random = Random(int(instance.id[2:]))

        references: Sequence[Reference] = instance.references
        if self.should_perturb_references:
            references = [self.perturb_reference(reference, rng) for reference in references]

        # Don't modify `id` of `Instance` here.
        # All the perturbed Instances generated from a single Instance should have the same ID.
        return replace(
            instance, input=self.perturb(instance.input, rng), references=references, perturbation=self.description,
        )

    def perturb_reference(self, reference: Reference, rng: Random) -> Reference:
        """Generates a new Reference by perturbing the output and tagging the Reference."""
        return replace(reference, output=self.perturb(reference.output, rng), tags=reference.tags + [self.name])

    @abstractmethod
    def perturb(self, text: str, rng: Random) -> str:
        """How to perturb the text. """
        pass


class PerturbationSpec(ObjectSpec):
    """Defines how to instantiate Perturbation."""

    pass


def create_perturbation(perturbation_spec: PerturbationSpec) -> Perturbation:
    """Creates Perturbation from PerturbationSpec."""
    return create_object(perturbation_spec)
