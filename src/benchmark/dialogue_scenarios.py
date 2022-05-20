import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from common.general import ensure_file_downloaded, format_text
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, EVAL_SPLITS
import pandas as pd
import json


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


@dataclass(frozen=True, eq=False)
class DialogueInstance(Instance):
    """
    A `DialogueInstance` is an Instance with additional dialogue specific
    attributes
    """

    # The first speaker's name
    initiator: Optional[Speaker] = None

    # The second speaker's name
    listener: Optional[Speaker] = None

    def render_lines(self) -> List[str]:
        info = super().render_lines()

        if self.initiator:
            info.append(f"initiator: {format_text(self.initiator.name)}")
        if self.listener:
            info.append(f"listener: {format_text(self.listener.name)}")

        return info


def get_whitelisted_prompts(path) -> List[str]:
    whitelist_df_raw: pd.DataFrame = pd.DataFrame(pd.read_csv(path, engine="python", index_col=0))

    # At each stage, the return value can be None, that's why the following if conditions
    # Makes it pass type checks and is generally good to do
    if whitelist_df_raw is not None:
        whitelist_df = whitelist_df_raw.dropna()
    if whitelist_df is not None:
        whitelist_df = whitelist_df.astype(dtype={"prompt": str, "sensitive": bool, "inappropriate": bool})
    if whitelist_df is not None:
        whitelist_df = whitelist_df.query("not inappropriate and not sensitive")
    assert whitelist_df is not None, "No whitelisted prompts left"

    return whitelist_df["prompt"].tolist()


def long_enough_prompt(prompt: str):
    """The prompt should be at least 20 characters"""
    return len(prompt) > 20


def is_whitelisted(prompt: str, whitelist: Set[str]):
    return prompt.lower() in whitelist


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

    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end

    def download_data(self):
        # Download the raw data
        self.data_path: str = os.path.join(self.output_path, "data")  # TODO: Ask Ashwin
        self.whitelist_path: str = os.path.join(self.output_path, "whitelisted_prompts.csv")
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/AshwinParanjape/"
            "whitelisted-dialogue-prompts/main/empatheticdialogues.csv",
            target_path=self.whitelist_path,
        )

    def read_instances(self):
        """Downloads the train, valid, test dataset and saves homogenized instances in self.instances"""

        instances: List[Instance] = []
        splits: Dict[str, str] = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}

        whitelisted_prompt_list = get_whitelisted_prompts(self.whitelist_path)
        whitelisted_prompt_list = [s.replace("_comma_", ",").strip().lower() for s in whitelisted_prompt_list]

        whitelisted_prompts: Set[str] = set(whitelisted_prompt_list[self.begin : self.end])
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
            data_df["utterance"] = (
                '<span class="conversation_utterance">"' + data_df["utterance"] + '"</span>'
            )  # Add HTML tags
            # Group rows by prompts, each group corresponds to an instance
            grouped_data_df = data_df.groupby(by=["prompt", "context"])
            for prompt_cols, prompt_df in grouped_data_df:
                prompt = prompt_cols[0]

                # For the test set, only use manually whitelisted prompts (for sensitivity)
                if splits[split] in EVAL_SPLITS and not is_whitelisted(prompt, whitelisted_prompts):
                    continue

                # The prompt should not be trivially short
                if not long_enough_prompt(prompt):
                    continue

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

                    # For training set (examples that will be used for in-context learning)
                    # use conversations with at least 6 turns (maximum length in the dataset)
                    if splits[split] == TRAIN_SPLIT and len(utterances) < 6:
                        continue

                    output = "".join([str(utt) for utt in utterances])
                    # Create a reference out of utterances
                    references.append(Reference(output=output, tags=[CORRECT_TAG],))

                # Should have at least one quality reference
                if splits[split] == TRAIN_SPLIT and len(references) == 0:
                    continue

                # Create an instance from multiple references
                instances.append(
                    Instance(
                        input=prompt_cols[0], references=references, split=splits[split], sub_split=prompt_cols[1],
                    )
                )
        return instances

    def get_instances(self) -> List[Instance]:
        self.download_data()
        return self.read_instances()


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
        self.data_path: str = os.path.join(
            self.output_path, "data/train_cleaned.json"
        )  # benchmark_output/scenarios/wizardofwikipedia/
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
        n_train = int(len(data_df) * 0.8)

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
                speakers = {
                    "0_Apprentice": apprentice_sp,
                    "1_Wizard": wizard_lis,
                    "0_Wizard": wizard_sp,
                    "1_Apprentice": apprentice_lis,
                }

                # Iterate through dialogue
                utterances = []
                dialog = row["dialog"]
                for utterance in dialog:
                    # Create Utterance object
                    utterance_text = '<span class="conversation_utterance">"' + utterance["text"] + '"</span>'
                    utterances.append(Utterance(speaker=speakers[utterance["speaker"]], text=utterance_text))

                # Join utterances into an output
                output = "".join([str(utt) for utt in utterances])

                # Create a reference from the output
                references = [Reference(output=output, tags=[CORRECT_TAG])]

                # Create an instance from reference (in our case, each instance should
                # have a single reference which corresponds to a "prompt")
                instances.append(Instance(input=topic, references=references, split=split))
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


class CommonSenseScenario(Scenario):
    """
    CommonSense-Dialogues dataset from this paper:

       https://arxiv.org/abs/2109.06427

    """

    name = "commonsense_dialogues"
    description = "A dataset of 11K dialogues grounded in social contexts involving utilization of commonsense."
    tags = ["interaction", "dialogue"]

    def __init__(self, begin: int, end: int, *args):
        self.begin = begin
        self.end = end

    def download_data(self):
        self.data_path: str = os.path.join(self.output_path, "data")
        self.whitelist_path: str = os.path.join(self.output_path, "whitelisted_prompts.csv")

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # Download the raw data
        train_path: str = os.path.join(self.data_path, "train.json")
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/alexa/Commonsense-Dialogues/main/data/train.json",
            target_path=train_path,
            unpack=False,
        )

        val_path: str = os.path.join(self.data_path, "valid.json")
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/alexa/Commonsense-Dialogues/main/data/valid.json",
            target_path=val_path,
            unpack=False,
        )

        test_path: str = os.path.join(self.data_path, "test.json")
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/alexa/Commonsense-Dialogues/main/data/test.json",
            target_path=test_path,
            unpack=False,
        )
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/AshwinParanjape/"
            "whitelisted-dialogue-prompts/main/commonsense_dialogues.csv",
            target_path=self.whitelist_path,
        )

    def _group_samples_by_context(self, samples: Dict[int, Dict]) -> Dict[str, List]:
        samples_by_context: Dict[str, List] = {}
        for idx, sample in samples.items():
            context = sample["context"]
            if context not in samples_by_context:
                samples_by_context[context] = []
            samples_by_context[context].append(sample)

        return samples_by_context

    def read_instances(self):
        """Downloads the train, valid, test dataset and saves homogenized instances in self.instances"""

        instances: List[Instance] = []
        splits: Dict[str, str] = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}

        whitelisted_prompt_list = get_whitelisted_prompts(self.whitelist_path)
        whitelisted_prompt_list = [s.replace("_comma_", ",").strip().lower() for s in whitelisted_prompt_list]

        whitelisted_prompts: Set[str] = set(whitelisted_prompt_list[self.begin : self.end])

        # Read the three splits
        for split in splits:
            file_path: str = os.path.join(self.data_path, f"{split}.json")
            if not os.path.exists(file_path):
                hlog(f"{file_path} doesn't exist, skipping")
                continue

            hlog(f"Reading {file_path}")

            with open(file_path) as fin:
                samples = json.load(fin)

            listener = Speaker(id=0, initiator=False, name="Bob")
            idx = 0
            for context, samps in self._group_samples_by_context(samples).items():
                if splits[split] in EVAL_SPLITS and not is_whitelisted(context, whitelisted_prompts):
                    continue

                references: List[Reference] = []

                for sample in samps:
                    # Skip over prompts from the eval set (valid and test) that aren't whitelisted
                    initiator = Speaker(id=int(idx) + 1, initiator=True, name=sample["speaker"])
                    idx += 1
                    utterances: List[Utterance] = []
                    for i, turn in enumerate(sample["turns"]):
                        if (i % 2) == 0:
                            speaker = initiator
                        else:
                            speaker = listener
                        utterance_text = '<span class="conversation_utterance">"' + turn + '"</span>'
                        utterances.append(Utterance(speaker=speaker, text=utterance_text))
                    output = "".join([str(utt) for utt in utterances])
                    references.append(Reference(output=output, tags=[CORRECT_TAG]))

                instances.append(
                    DialogueInstance(
                        input=context,
                        references=references,
                        split=splits[split],
                        initiator=initiator,
                        listener=listener,
                    )
                )

        return instances

    def get_instances(self) -> List[Instance]:
        self.download_data()
        return self.read_instances()


if __name__ == "__main__":
    # Test EmpatheticDialoguesScenario
    """
    scenario = EmpatheticDialoguesScenario()
    scenario.output_path = "./benchmark_output/scenarios/empatheticdialogues"
    instances = scenario.get_instances()
    print(len(instances))
    print(instances[100])

    # Test WizardOfWikipediaScenario
    scenario = WizardOfWikipediaScenario()
    scenario.output_path = "./benchmark_output/scenarios/wizardofwikipedia"
    instances = scenario.get_instances()
    print(len(instances))
    print(instances[100])
    """
    # Test CommonSenseScenario
    scenario = CommonSenseScenario(begin=0, end=10)
    scenario.output_path = "./benchmark_output/scenarios/commonsense_dialogues"
    instances = scenario.get_instances()
    print(len(instances))
    print(instances[100])
