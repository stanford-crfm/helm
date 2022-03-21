import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG
from benchmark.augmentations.perturbation_description import PerturbationDescription
import pandas as pd


@dataclass
class Speaker:
    id: int
    initiator: bool
    name: Optional[str] = None


@dataclass
class Utterance:
    speaker: Speaker
    text: str


class DialogueReference(Reference):
    """
    A reference here specifies
    """

    # The output text
    output: List[Utterance]

    # Extra metadata (e.g., whether it's correct/factual/toxic)
    tags: List[str]

    @property
    def is_correct(self) -> bool:
        return True

    def render_lines(self) -> List[str]:
        return [f"reference {format_tags(self.tags)}: {format_text(self.output)}"]


@dataclass(frozen=True)
class DialogueInstance(Instance):
    """
    A dialogue instance is uniquely identified by the `input` which stores the
    "prompt"/"content"/"situation" provided as part of the dialogue dataset.
    Multiple human conversations collected with the same `input` are stored in `references`
    as a list of `DialogueReference`
    """

    # The input text
    input: str

    # References that helps us evaluate
    references: List[DialogueReference]

    # Split (e.g., train, valid, test)
    split: Optional[str] = None

    # Sub split (e.g. toxic, non-toxic)
    sub_split: Optional[str] = None

    # Used to group Instances that were created from a particular Instance through data augmentation
    id: Optional[str] = None

    # Description of the Perturbation that was applied when creating this Instance
    perturbation: Optional[PerturbationDescription] = None

    # metadata about the dialogue instance (scenario specific)
    metadata: Optional[Dict] = None

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


class EmpatheticDialoguesScenario(Scenario):
    """
    The Empathetic Dialogues dataset from this paper:

        https://arxiv.org/abs/1811.00207

    Code is adapted from:

       https://github.com/facebookresearch/EmpatheticDialogues
    """

    name = "empatheticdialogues"
    description = "A dataset of 25k conversations grounded in emotional situations."
    tags = ["interaction", "dialogue"]

    def __init__(self, *args):
        pass

    def prepare_instances(self):
        """Downloads the train, valid, test dataset and saves homogenized instances in self.instances"""

        # Download the raw data
        data_path: str = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
            target_path=data_path,
            unpack=True,
        )

        instances: List[Instance] = []
        splits: Dict[str, str] = {"train": TRAIN_SPLIT}  # , "valid": VALID_SPLIT, "test": TEST_SPLIT}

        # Read the three splits
        for split in splits:
            csv_path: str = os.path.join(data_path, f"{split}.csv")
            if not os.path.exists(csv_path):
                hlog(f"{csv_path} doesn't exist, skipping")
                continue

            hlog(f"Reading {csv_path}")

            column_names = "conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags".split(",")
            data_df = pd.read_csv(
                csv_path, engine="python", quoting=csv.QUOTE_NONE, header=0, names=column_names, index_col=False
            )

            # Reformat dataset idiosyncracies
            data_df["prompt"] = data_df["prompt"].str.replace("_comma_", ",").str.strip()
            data_df["utterance"] = data_df["utterance"].str.replace("_comma_", ",").str.strip()

            # Group rows by prompts, each group corresponds to an instance
            grouped_data_df = data_df.groupby(by=["prompt", "context"])
            for prompt_cols, prompt_df in grouped_data_df:

                # Group rows by conversations, each group corresponds to a reference
                grouped_prompt_df = prompt_df.groupby(["conv_id", "selfeval"])
                references = []
                for conv_cols, conv_df in grouped_prompt_df:
                    if len(conv_df) < 2:
                        continue  # Filter conversations with only one row
                    grouped_df = conv_df.sort_values("utterance_idx")
                    initiator = Speaker(id=grouped_df.iloc[0]["speaker_idx"], initiator=True)
                    listener = Speaker(id=grouped_df.iloc[1]["speaker_idx"], initiator=False)
                    speakers = {initiator.id: initiator, listener.id: listener}

                    # Create a list of utterances
                    utterances = [
                        Utterance(speaker=speakers[row["speaker_idx"]], text=row["utterance"])
                        for idx, row in grouped_df.iterrows()
                    ]

                    # Create a reference out of utterances
                    references.append(DialogueReference(output=utterances, tags=[]))

                # Create an instance from multiple references

                instances.append(
                    DialogueInstance(
                        input=prompt_cols[0],
                        references=references,
                        split=splits[split],
                        metadata={"emotion": prompt_cols[1]},
                    )
                )
        return instances

    def filter_instances(self, instances):
        """Applies following filters to select instances from self.instances"""

        # reference conversation length filter
        # instances = [i for i in instances if i.references]

        return instances

    def get_instances(self) -> List[Instance]:
        return self.filter_instances(self.prepare_instances())
        return self.instances


if __name__ == "__main__":
    scenario = EmpatheticDialoguesScenario()
    scenario.output_path = "../benchmark_output/scenarios/empatheticdialogues"
    instances = scenario.get_instances()
    print(instances[100])
