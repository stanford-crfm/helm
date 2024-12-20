import csv
import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


@dataclass(frozen=True)
class Speaker:
    id: int
    initiator: bool
    name: Optional[str] = None


@dataclass(frozen=True)
class Utterance:
    speaker: Speaker
    text: str

    def __str__(self):
        assert self.speaker.name is not None, "Speaker name needs to be set for generating utterance string"
        return f"{self.speaker.name}: {self.text}" + "\n"


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
        super().__init__()
        pass

    def download_data(self, output_path: str):
        # Download the raw data
        self.data_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url="https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
            target_path=self.data_path,
            unpack=True,
        )

    def read_instances(self):
        """Downloads the train, valid, test dataset and saves homogenized instances in self.instances"""

        instances: List[Instance] = []
        splits: Dict[str, str] = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}

        # Read the three splits
        for split in splits:
            csv_path: str = os.path.join(self.data_path, f"{split}.csv")
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
                        continue  # Filter conversations without 2 speaker utterances
                    grouped_df = conv_df.sort_values("utterance_idx")
                    initiator = Speaker(id=grouped_df.iloc[0]["speaker_idx"], initiator=True, name="Bob")
                    listener = Speaker(id=grouped_df.iloc[1]["speaker_idx"], initiator=False, name="Jen")
                    speakers = {initiator.id: initiator, listener.id: listener}

                    # Create a list of utterances
                    utterances = [
                        Utterance(speaker=speakers[row["speaker_idx"]], text=row["utterance"].strip())
                        for idx, row in grouped_df.iterrows()
                    ]

                    output: str = "".join([str(utt) for utt in utterances])
                    # Create a reference out of utterances
                    references.append(Reference(Output(text=output), tags=[CORRECT_TAG]))

                # Create an instance from multiple references

                instances.append(
                    Instance(
                        input=Input(text=prompt_cols[0]),
                        references=references,
                        split=splits[split],
                        sub_split=prompt_cols[1],
                    )
                )
        return instances

    def filter_instances(self, instances):
        """Applies following filters to select instances from self.instances"""

        # TODO: Write code to only keep the better instances for few shot prompting
        # The given prompt is too short
        # def short_prompt(instance: Instance): return len(instance.input)<20

        # The conversation has less than 6 utterances (6 is max)
        # def short_convo(reference: DialogueReference): return len(reference.output)<6

        # The conversation has less than 6 utterances (6 is max)
        # def short_convo(instance: Instance): return len(instance.references)<6

        # instances = filter()
        # reference conversation length filter
        # instances = [i for i in instances if i.references]

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        self.download_data(output_path)
        return self.filter_instances(self.read_instances())


if __name__ == "__main__":
    scenario = EmpatheticDialoguesScenario()
    instances = scenario.get_instances(os.path.join(get_benchmark_output_path(), "scenarios/empatheticdialogues"))
    print(instances[100])
