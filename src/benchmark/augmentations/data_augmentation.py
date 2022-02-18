from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from benchmark.scenario import Instance, Reference
from common.object_spec import ObjectSpec, create_object
from dacite import from_dict


class DataAugmentation(ABC):

    # Unique name to describe data augmentation. We use the name to tag instances.
    name: str

    def apply(self, id_tag: str, instance: Instance, should_perturb_references: bool = True) -> Instance:
        """Perturbs input, tags the instance and perturbs the references, if should_perturb_references is true."""
        perturbed_instance = asdict(instance)
        perturbed_instance["input"] = self.perturb(perturbed_instance["input"])

        # Tag the instance with the unique id and the name of the data augmentation
        perturbed_instance["tags"].extend([id_tag, self.tag])

        if should_perturb_references:
            perturbed_instance["references"] = [self.perturb_reference(reference) for reference in instance.references]

        return from_dict(Instance, perturbed_instance)

    def perturb_reference(self, reference: Reference) -> Reference:
        """Perturb a Reference."""
        perturbed_reference = asdict(reference)
        perturbed_reference["output"] = self.perturb(perturbed_reference["output"])
        perturbed_reference["tags"].append(self.tag)
        return from_dict(Reference, perturbed_reference)

    @abstractmethod
    def perturb(self, text: str) -> str:
        """How to perturb the text. """
        pass

    @property
    def tag(self) -> str:
        """Used to tag instances to indicate what data augmentation has been applied."""
        return self.name


class DataAugmentationSpec(ObjectSpec):
    """Defines how to instantiate DataAugmentation."""

    pass


def create_data_augmentation(data_augmentation_spec: DataAugmentationSpec) -> DataAugmentation:
    """Creates DataAugmentation from DataAugmentationSpec."""
    return create_object(data_augmentation_spec)


# TODO: Get rid of this after we add the new instance fields:
#       https://github.com/stanford-crfm/benchmarking/issues/124
@dataclass
class CleanAugmentation(DataAugmentation):
    """Doesn't apply data augmentation, but just adds 'clean' to the list of tags."""

    CLEAN_TAG = "clean"

    name = CLEAN_TAG

    def perturb(self, text: str) -> str:
        return text


@dataclass
class ExtraSpaceAugmentation(DataAugmentation):
    """A toy data augmentation that adds additional spaces to existing spaces."""

    name = "extra_space"

    def __init__(self, num_spaces: int):
        self.num_spaces = num_spaces

    def perturb(self, text: str) -> str:
        return text.replace(" ", " " * self.num_spaces)

    @property
    def tag(self) -> str:
        return f"{self.name}|num_spaces={self.num_spaces}"
