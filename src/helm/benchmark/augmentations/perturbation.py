from abc import ABC, abstractmethod
from dataclasses import replace
from random import Random
from typing import List, Optional


from .perturbation_description import PerturbationDescription
from helm.benchmark.scenarios.scenario import Input, Instance, Reference, Output
from helm.common.object_spec import ObjectSpec, create_object


class Perturbation(ABC):

    # Unique name to describe perturbation
    name: str

    # Whether to perturb references
    should_perturb_references: bool = False

    @property
    def description(self) -> PerturbationDescription:
        """Description of the perturbation."""
        return PerturbationDescription(name=self.name)

    def get_rng(self, instance: Instance, seed: Optional[int] = None) -> Random:
        """Creates a random number generator using the `Instance` id and `seed`."""
        assert instance.id is not None
        # If seed exists, use it as part of the random seed
        return Random(instance.id if seed is None else str(seed) + instance.id)

    def apply(self, instance: Instance, seed: Optional[int] = None) -> Instance:
        """
        Generates a new Instance by perturbing the input, tagging the Instance and perturbing the References,
        if should_perturb_references is true. Initializes a random number generator based on instance_id that gets
        passed to perturb and perturb_references.
        """
        rng: Random = self.get_rng(instance, seed)

        references: List[Reference] = instance.references
        if self.should_perturb_references:
            references = [self.perturb_reference(reference, rng) for reference in references]

        description = replace(self.description, seed=seed)

        # Don't modify `id` of `Instance` here.
        # All the perturbed Instances generated from a single Instance should have the same ID.
        return replace(
            instance,
            input=Input(text=self.perturb(instance.input.text, rng)),
            references=references,
            perturbation=description,
        )

    def perturb_reference(self, reference: Reference, rng: Random) -> Reference:
        """Generates a new Reference by perturbing the output and tagging the Reference."""
        return replace(reference, output=Output(text=self.perturb(reference.output.text, rng)), tags=reference.tags)

    @abstractmethod
    def perturb(self, text: str, rng: Random) -> str:
        """How to perturb the text."""
        pass


class PerturbationSpec(ObjectSpec):
    """Defines how to instantiate Perturbation."""

    pass


def create_perturbation(perturbation_spec: PerturbationSpec) -> Perturbation:
    """Creates Perturbation from PerturbationSpec."""
    return create_object(perturbation_spec)
