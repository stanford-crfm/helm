import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG
import pandas as pd


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
        pass

    def download_data(self):
        # Download the raw data
        self.data_path: str = os.path.join(self.output_path, "data/empatheticdialogues/") #TODO: Ask Ashwin
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

                    output = "".join([str(utt) for utt in utterances])
                    # Create a reference out of utterances
                    references.append(Reference(output=output, tags=[CORRECT_TAG],))

                # Create an instance from multiple references

                instances.append(
                    Instance(
                        input=prompt_cols[0], references=references, split=splits[split], sub_split=prompt_cols[1],
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

    def get_instances(self) -> List[Instance]:
        self.download_data()
        return self.filter_instances(self.read_instances())

class WizardOfWikipediaScenario(Scenario):
    """
    The Wizard of Wikipedia dataset from this paper:

       https://arxiv.org/abs/1811.01241

    """

    name = "wizardofwikipedia"
    description = "A dataset of 25k conversations grounded in emotional situations."
    tags = ["interaction", "dialogue"]

    def __init__(self, *args):
        pass

    def download_data(self):
        # Download the raw data
        self.data_path: str = os.path.join(self.output_path, "data/train_cleaned.json") # benchmark_output/scenarios/wizardofwikipedia/
        ensure_file_downloaded(
            source_url="https://dialogue-benchmark-wow.s3.amazonaws.com/train_pared.json",
            target_path=self.data_path,
            unpack=False,
        )

    def read_instances(self):
        """Downloads the train, valid, test dataset and saves homogenized instances in self.instances"""

        instances: List[Instance] = []

        hlog(f"Reading {self.data_path}")

        data_df = pd.read_json(self.data_path)
        n_train = int(len(data_df)*0.8)

        # Initialize references
        references = []

        # Split df and iterate through splits
        train_df, test_df = data_df.iloc[:n_train], data_df.iloc[n_train:]
        split_to_df = {TRAIN_SPLIT: train_df, TEST_SPLIT: test_df}

        # Iterate through conversations in the df
        for split in split_to_df.keys():
            df = split_to_df[split]
            for index, row in df.iterrows():

                # Check Wizard quality and skip if wizard_eval < 5
                if row["wizard_eval"] < 5:
                    continue

                # Cache topic
                topic = row["chosen_topic"]

                # Create Speakers
                apprentice_sp = Speaker(id=0, initiator=True, name="Bob")
                apprentice_lis = Speaker(id=1, initiator=False, name="Bob")
                wizard_sp = Speaker(id=2, initiator=True, name="Jen")
                wizard_lis = Speaker(id=3, initiator=False, name="Jen")
                speakers = {"0_Apprentice": apprentice_sp, "1_Wizard": wizard_lis,
                            "0_Wizard": wizard_sp, "1_Apprentice": apprentice_lis}
                
                # Iterate through dialogue
                utterances = [] 
                dialog = row["dialog"]
                for utterance in dialog:
                    # Create Utterance object
                    utterances.append(Utterance(speaker=speakers[utterance["speaker"]], text=utterance["text"]))
                    
                # Join utterances into an output
                output = "".join([str(utt) for utt in utterances])

                # Create a reference from the output
                references = [Reference(output=output, tags=[CORRECT_TAG])]

                # Create an instance from reference (in our case, each instance should 
                # have a single reference which corresponds to a "prompt")
                instances.append(
                        Instance(
                            input=topic, references=references, split=split
                        )
                    )
        return instances

    def filter_instances(self, instances):
        """Applies following filters to select instances from self.instances"""

        # TODO: Write code to only keep the better instances for few shot prompting
        # The given prompt is too short or prompt is the number 1
        # def short_prompt(instance: Instance): return len(instance.input)<20

        # The conversation has less than 6 utterances (6 is max)
        # def short_convo(reference: DialogueReference): return len(reference.output)<6

        # The conversation has less than 6 utterances (6 is max)
        # def short_convo(instance: Instance): return len(instance.references)<6

        # instances = filter()
        # reference conversation length filter
        # instances = [i for i in instances if i.references]

        return instances

    def get_instances(self) -> List[Instance]:
        self.download_data()
        return self.filter_instances(self.read_instances())

if __name__ == "__main__":
    # Test EmpatheticDialoguesScenario
    """
    scenario = EmpatheticDialoguesScenario()
    scenario.output_path = "./benchmark_output/scenarios/empatheticdialogues"
    instances = scenario.get_instances()
    print(len(instances))
    print(instances[100])
    """
    # Test WizardOfWikipediaScenario
    scenario = WizardOfWikipediaScenario()
    scenario.output_path = "./benchmark_output/scenarios/wizardofwikipedia"
    instances = scenario.get_instances()
    print(len(instances))
    print(instances[100])