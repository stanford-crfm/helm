import json
import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Reference, Scenario, Instance, Input, TEST_SPLIT, Output


class LMEntryScenario(Scenario):
    """
    The LMentry Benchmark
    https://arxiv.org/pdf/2211.02069.pdf

    The implementation is with reference to the original repo: https://github.com/aviaefrat/lmentry
    The data is also downloaded from the repo.

    LMentry evaluates LM's abilities of performing elementary language tasks. Examples include
    finding which word is shorter, or which word is the last in a sentence.
    """

    name = "lm_entry"
    description = "The LMentry benchmark for elementary language tasks"
    tags: List[str] = []
    url_template = "https://raw.githubusercontent.com/aviaefrat/lmentry/main/data/{subtask}.json"
    task_to_subtasks = {
        "all_words_from_category": [
            "all_words_from_category",
            "all_words_from_category_0_distractors",
            "all_words_from_category_1_distractors",
            "all_words_from_category_2_distractors",
        ],
        "any_words_from_category": [
            "any_words_from_category",
            "any_words_from_category_3_distractors",
            "any_words_from_category_4_distractors",
            "any_words_from_category_5_distractors",
        ],
        "bigger_number": ["bigger_number"],
        # "ends_with_letter": ["ends_with_letter"], # HELM's metrics currently don't support this
        # "ends_with_word": ["ends_with_word"], # HELM's metrics currently don't support this
        "first_alphabetically": [
            "first_alphabetically",
            "first_alphabetically_consecutive_first_letter",
            "first_alphabetically_different_first_letter",
            "first_alphabetically_far_first_letter",
            "first_alphabetically_same_first_letter",
        ],
        "first_letter": ["first_letter"],
        "first_word": ["first_word"],
        "homophones": ["homophones"],
        "last_letter": ["last_letter"],
        "last_word": ["last_word"],
        "least_associated_word": ["least_associated_word"],
        "less_letters": ["less_letters", "less_letters_length_diff_1", "less_letters_length_diff_3plus"],
        "more_letters": ["more_letters", "more_letters_length_diff_1", "more_letters_length_diff_3plus"],
        "most_associated_word": ["most_associated_word"],
        "rhyming_word": [
            "rhyming_word",
            "rhyming_word_orthographically_different",
            "rhyming_word_orthographically_similar",
        ],
        # "sentence_containing": ["sentence_containing"], # HELM's metrics currently don't support this
        # "sentence_not_containing": ["sentence_not_containing"], # HELM's metrics currently don't support this
        "smaller_number": ["smaller_number"],
        # "starts_with_letter": ["starts_with_letter"], # HELM's metrics currently don't support this
        # "starts_with_word": ["starts_with_word"], # HELM's metrics currently don't support this
        "word_after": ["word_after"],
        "word_before": ["word_before"],
        # "word_containing": ["word_containing"], # HELM's metrics currently don't support this
        # "word_not_containing": ["word_not_containing"], # HELM's metrics currently don't support this
    }

    def __init__(self, task: str):
        super().__init__()
        assert task in self.task_to_subtasks, f"Unsupported task: {task}"
        self.task: str = task

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_paths: List[str] = []

        for subtask in self.task_to_subtasks[self.task]:
            data_path: str = os.path.join(output_path, f"{subtask}.json")
            ensure_file_downloaded(
                source_url=self.url_template.format(subtask=subtask),
                target_path=data_path,
                unpack=False,
            )
            data_paths.append(data_path)

        def generate_references_for_multiple_choice_question(
            options: List[str], correct_answer: str
        ) -> List[Reference]:
            references: List[Reference] = []

            for option in options:
                if option == correct_answer:
                    references.append(Reference(Output(text=option), tags=[CORRECT_TAG]))
                else:
                    references.append(Reference(Output(text=option), tags=[]))

            return references

        def generate_references_for_generation_question(correct_answer: str) -> List[Reference]:
            return generate_references_for_multiple_choice_question(
                options=[correct_answer], correct_answer=correct_answer
            )

        instances: List[Instance] = []

        for data_path in data_paths:
            data: dict = json.load(open(data_path))

            for example in data["examples"].values():
                input_text: str = example["input"]

                # Normalize the input text to the same format
                if input_text.startswith("Q: "):
                    input_text = input_text[3:]

                if input_text.endswith("\n"):
                    input_text = input_text[:-1]
                elif input_text.endswith("\nA:"):
                    input_text = input_text[:-3]

                input = Input(text=input_text)
                references: List[Reference]

                if self.task == "all_words_from_category":
                    correct_answer = "yes" if example["metadata"]["num_distractors"] == 0 else "no"
                    references = generate_references_for_multiple_choice_question(
                        options=["yes", "no"], correct_answer=correct_answer
                    )
                elif self.task == "any_words_from_category":
                    correct_answer = "yes" if len(example["metadata"]["category_words"]) > 0 else "no"
                    references = generate_references_for_multiple_choice_question(
                        options=["yes", "no"], correct_answer=correct_answer
                    )
                elif self.task == "bigger_number" or self.task == "smaller_number":
                    references = generate_references_for_multiple_choice_question(
                        options=[str(example["metadata"]["n1"]), str(example["metadata"]["n2"])],
                        correct_answer=str(example["metadata"]["answer"]),
                    )
                elif self.task == "first_alphabetically":
                    references = generate_references_for_multiple_choice_question(
                        options=[example["metadata"]["word1"], example["metadata"]["word2"]],
                        correct_answer=example["metadata"]["answer"],
                    )
                elif self.task == "first_letter" or self.task == "last_letter":
                    references = generate_references_for_generation_question(example["metadata"]["answer"])
                elif self.task == "first_word" or self.task == "last_word":
                    references = generate_references_for_generation_question(example["metadata"]["answer"])
                elif self.task == "homophones":
                    references = generate_references_for_multiple_choice_question(
                        options=[example["metadata"]["answer"], example["metadata"]["distractor"]],
                        correct_answer=example["metadata"]["answer"],
                    )
                elif self.task == "least_associated_word" or self.task == "most_associated_word":
                    references = generate_references_for_multiple_choice_question(
                        options=[example["metadata"]["answer"]] + example["metadata"]["distractors"],
                        correct_answer=example["metadata"]["answer"],
                    )
                elif self.task == "less_letters" or self.task == "more_letters":
                    references = generate_references_for_multiple_choice_question(
                        options=[example["metadata"]["word1"], example["metadata"]["word2"]],
                        correct_answer=example["metadata"]["answer"],
                    )
                elif self.task == "rhyming_word":
                    references = generate_references_for_multiple_choice_question(
                        options=[example["metadata"]["answer"], example["metadata"]["distractor"]],
                        correct_answer=example["metadata"]["answer"],
                    )
                elif self.task == "word_before" or self.task == "word_after":
                    references = generate_references_for_generation_question(example["metadata"]["answer"])
                else:
                    raise ValueError(f"Unsupported task: {self.task}")

                instance = Instance(
                    input=input,
                    references=references,
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        return instances
