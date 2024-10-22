from abc import ABC, abstractmethod
from dataclasses import replace
from random import Random
from typing import List, Optional


from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
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

    @abstractmethod
    def apply(self, instance: Instance, seed: Optional[int] = None) -> Instance:
        """Generate a modified instance from the input instance."""
        pass


class TextPerturbation(Perturbation, ABC):
    def apply(self, instance: Instance, seed: Optional[int] = None) -> Instance:
        """
        Generates a new Instance by applying `perturb` to the input and (if requested) the references.
        Initializes a random number generator based on instance_id that gets
        passed to perturb.
        """
        rng: Random = self.get_rng(instance, seed)

        references: List[Reference] = instance.references
        if self.should_perturb_references:
            references = [self._perturb_reference(reference, rng) for reference in references]

        description = replace(self.description, seed=seed)

        perturbed_input: Input
        if instance.input.multimedia_content:
            perturbed_media_objects = []
            for media_object in instance.input.multimedia_content.media_objects:
                # Apply perturbations to the text data of the multimedia content
                if media_object.is_type("text") and media_object.text is not None:
                    perturbed_media_objects.append(replace(media_object, text=self.perturb(media_object.text, rng)))
                else:
                    perturbed_media_objects.append(media_object)

            perturbed_input = Input(
                multimedia_content=replace(instance.input.multimedia_content, media_objects=perturbed_media_objects)
            )
        else:
            perturbed_input = Input(text=self.perturb(instance.input.text, rng))

        # Don't modify `id` of `Instance` here.
        # All the perturbed Instances generated from a single Instance should have the same ID.
        return replace(
            instance,
            input=perturbed_input,
            references=references,
            perturbation=description,
            contrast_inputs=[instance.input],
        )

    def _perturb_reference(self, reference: Reference, rng: Random) -> Reference:
        """Generates a new Reference by perturbing the output and tagging the Reference."""
        return replace(
            reference,
            output=Output(
                text=self.perturb(reference.output.text, rng), multimedia_content=reference.output.multimedia_content
            ),
            tags=reference.tags,
        )

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
