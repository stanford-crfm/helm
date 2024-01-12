import os
import json
from typing import List, Dict
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Reference,
    Output,
    CORRECT_TAG,
)

TASK = {
    "knowledge": ["qa_2020", "qa_2023"],
    "style": [
        "base",
        "shake_w",
        "augment",
        "shake_p0",
        "shake_p0.6",
        "bible_p0",
        "bible_p0.6",
        "romantic_p0",
        "romantic_p0.6",
        "tweet_p0",
        "tweet_p0.6",
    ],
}
DEMO = {
    "qa_2020": ["2021_qa", "global_facts", "machine_learning", "moral_scenarios", "us_foreign_policy"],
    "base": ["base"],
    "shake_p0.6": ["base", "shake_p0.6"],
    "bible_p0.6": ["base", "bible_p0.6"],
    "romantic_p0.6": ["base", "romantic_p0.6"],
    "tweet_p0.6": ["base", "tweet_p0.6"],
}

DATA_REPO_HASH = "38972f6ccbf376a8d0660babafb4d2b3b9cca3f4"


class DecodingTrustOODRobustnessScenario(Scenario):
    """
    This scenario is based on the OOD robustness section (Section 6) of the DecodingTrust benchmark To evaluate the
    OOD robustness, we construct three different scenarios, which we investigate 1) OOD style, with different style
    transformations from original SST-2 test set 2) OOD knowledge, QA2020 and QA2023 from RealtimeQA with standard
    and 'I don't know' setting, which we aim to evaluate the reliability of models when facing questions beyond date
    of training data 3) OOD in-context demonstrations, we evaluate two different OOD demonstrations,
    namely demonstrations with different domains and styles.
    """

    name: str = "decodingtrust_ood"
    description: str = "Out-of-distribution Robustness"
    tags: List[str] = ["robustness", "trustworthy"]
    source_url: str = (
        f"https://raw.githubusercontent.com/AI-secure/DecodingTrust-Data-Legacy/{DATA_REPO_HASH}/data/ood/{{}}.json"
    )

    def __init__(self, ood_type: str, task: str, demo_name: str, run_id: int, idk: bool) -> None:
        """
        Args:
          ood_type: which OOD aspect to be evaluated
          task: specific OOD task to perform
          demo_name: when evaluating OOD in-context demonstrations, which type of OOD demonstrations should be used
          run_id: specific subset of selected type of OOD demonstrations
          idk: when evaluating OOD knowledge, if we want to use "I don't know" mode.
        """
        super().__init__()

        if ood_type not in TASK.keys():
            raise ValueError("Invalid OoD type name specified!")
        if task not in TASK[ood_type]:
            raise ValueError("Invalid OoD task name specified!")
        if 0 <= run_id < 3:
            if demo_name not in DEMO[task]:
                raise ValueError("Invalid OoD demo name specified!")

        self.ood_type = ood_type
        self.task = task
        self.demo_name = demo_name
        self.run_id = run_id
        self.idk = idk
        self.few_shot_num = 0
        if 0 <= run_id < 3:
            if ood_type == "style":
                self.few_shot_num = 8
            elif ood_type == "knowledge":
                self.few_shot_num = 5  # 5 in our ood dataset
        self.task_message = None

    def get_task_messages(self):
        if self.ood_type == "knowledge":
            task_message = None
        elif self.ood_type == "style":
            task_message = (
                "For each snippet of text, label the sentiment of the text as positive or negative. "
                "The answer should be exactly 'positive' or 'negative'."
            )
        else:
            raise ValueError
        return task_message

    def get_demonstrations(self, datasets, demo):
        dataset_demo: List[List] = []
        dataset_example: List[Dict[str, str]] = []
        if self.few_shot_num == 0:
            return dataset_demo
        if self.ood_type == "knowledge":
            dataset_example = datasets["demo"][demo]
            for index in range(self.few_shot_num):
                context_sentences = ""
                if "2021_qa" in demo:
                    context_sentences = "Today is {}. \n".format(dataset_example[index]["question_date"])
                question = dataset_example[index]
                context_sentences += "Question: " + question["question_sentence"] + "\n"
                dataset_demo.append([context_sentences, int(question["answer"][0]), question["choices"]])
        elif self.ood_type == "style":
            dataset_example = datasets["train_demo"][demo]
            labels = ["negative", "positive"]
            for i in range(self.few_shot_num):
                example = dataset_example[i]
                dataset_demo.append([example["sentence"], labels[int(example["label"])], labels])
        return dataset_demo

    def get_input(self, text) -> Input:
        if self.task_message is not None:
            message = f"{self.task_message}\n{text}"
        else:
            message = text

        return Input(text=message)

    def get_references(self, label: str, answer_mapping: List[str]) -> List[Reference]:
        references: List[Reference] = []
        for candidate_label, label_text in enumerate(answer_mapping):
            if self.ood_type == "style":
                tags = [CORRECT_TAG] if label == label_text else []
                references.append(Reference(output=Output(text=label_text), tags=tags))
            elif self.ood_type == "knowledge":
                tags = [CORRECT_TAG] if label == candidate_label else []
                references.append(Reference(output=Output(text=label_text), tags=tags))
        return references

    def get_prompts(self, datasets):
        dataset = []
        if self.ood_type == "knowledge":
            dataset_question = datasets["test"][self.task]
            for index in range(len(dataset_question)):
                question = dataset_question[index]
                answer = question["answer"][0]
                context_sentences = "Today is {}. \n".format(question["question_date"])
                choices = question["choices"]
                # Add the context text sentences and choices to the item
                context_sentences += "Question: " + question["question_sentence"] + "\n"
                if self.idk:
                    choices.append("I don't know")
                dataset.append({"input": context_sentences, "label": int(answer), "option": choices})
        elif self.ood_type == "style":
            dataset_question = datasets["dev"][self.task]
            labels = ["negative", "positive"]
            for idx, example in enumerate(dataset_question):
                context_sentences = ""
                context_sentences += example["sentence"]
                label = labels[int(example["label"])]
                dataset.append({"input": context_sentences, "label": label.lower(), "option": labels})
        return dataset

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, f"{self.ood_type}.json".replace("/", "_"))

        ensure_file_downloaded(
            source_url=self.source_url.format(self.ood_type),
            target_path=data_path,
        )

        with open(data_path) as f:
            dataset: List[Dict] = json.load(f)

        self.task_message = self.get_task_messages()
        curr_demo_name = self.demo_name + "_" + str(self.run_id)
        task_demo = self.get_demonstrations(dataset, curr_demo_name)
        processed_dataset = self.get_prompts(dataset)

        instances: List[Instance] = []
        for raw_item in task_demo:
            instance = Instance(
                input=self.get_input(text=raw_item[0]),
                references=self.get_references(raw_item[1], raw_item[2]),
                split=TRAIN_SPLIT,
                sub_split=self.demo_name,
            )
            instances.append(instance)
        for raw_item in processed_dataset:
            instance = Instance(
                input=self.get_input(text=raw_item["input"]),
                references=self.get_references(raw_item["label"], raw_item["option"]),
                split=VALID_SPLIT,
                sub_split=self.demo_name,
            )
            instances.append(instance)

        return instances
