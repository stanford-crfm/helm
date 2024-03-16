import argparse
import csv
import json
import os
import random
import shutil
import requests
import statistics
from typing import Any, Dict, List

from tqdm import tqdm

from helm.common.hierarchical_logger import hlog, htrack_block

"""
Script to create the HEIM Human Evaluation dataset for reproducibility of the dataset.

python3 scripts/heim_human_eval.py <Path to raw annotations>
"""

# Fix seed
random.seed(0)


QUESTION_TYPE_TO_INFOS = {
    "alignment": {
        "instruction": "Please answer the question below about the following image and description.",
        "question": "How well does the image match the description?",
        "choices": {
            0: "Does not match at all",
            1: "Has significant discrepancies",
            2: "Has several minor discrepancies",
            3: "Has a few minor discrepancies",
            4: "Matches exactly",
        },
        "template": "{instruction}\n{image}\nDescription: {prompt}\n\n{question}\n{choices}",
    },
    "clear_subject": {
        "instruction": "Please answer the question below about the following image and description.",
        "question": "Is it clear who the subject(s) of the image is? The subject can be a living being "
        "(e.g., a dog or a person) or an inanimate body or object (e.g., a mountain).",
        "choices": {
            0: "No, it's unclear.",
            1: "I don't know. It's hard to tell.",
            2: "Yes, it's clear.",
        },
        "template": "{instruction}\n{image}\nDescription: {prompt}\n\n{question}\n{choices}",
    },
    "aesthetics": {
        "instruction": "Please answer the question below about the following image and description.",
        "question": "How aesthetically pleasing is the image?",
        "choices": {
            0: "I find the image ugly.",
            1: "The image has a lot of flaws, but it's not completely unappealing.",
            2: "I find the image neither ugly nor aesthetically pleasing.",
            3: "The image is aesthetically pleasing and nice to look at it.",
            4: "The image is aesthetically stunning. I can look at it all day.",
        },
        "template": "{instruction}\n{image}\nDescription: {prompt}\n\n{question}\n{choices}",
    },
    "originality": {
        "instruction": "Please answer the question below about the following image and description.",
        "question": "How original is the image, given it was created with the description?",
        "choices": {
            0: "I've seen something like this before to the point it's become tiresome.",
            1: "The image is not really original, but it has some originality to it.",
            2: "Neutral.",
            3: "I find the image to be fresh and original.",
            4: "I find the image to be extremely creative and out of this world.",
        },
        "template": "{instruction}\n{image}\nDescription: {prompt}\n\n{question}\n{choices}",
    },
    "photorealism": {
        "instruction": "Determine if the following image is AI-generated or real.",
        "question": "Does the image look like an AI-generated photo or a real photo?",
        "choices": {
            0: "AI-generated photo",
            1: "Probably an AI-generated photo, but photorealistic",
            2: "Neutral",
            3: "Probably a real photo, but with irregular textures and shapes",
            4: "Real photo",
        },
        "template": "{instruction}\n{image}\n{question}\n{choices}",
    },
}

QUESTION_TYPES_TO_FOLDERS: Dict[str, List[str]] = {
    "alignment": [
        "vhelm_image_critique_aesthetics",
        "vhelm_image_critique_originality_subject_aesthetics",
        "vhelm_image_critique_subject_aesthetics",
        "vhelm_image_critique",
    ],
    "clear_subject": ["vhelm_image_critique_originality_subject_aesthetics", "vhelm_image_critique_subject_aesthetics"],
    "aesthetics": [
        "vhelm_image_critique_aesthetics",
        "vhelm_image_critique_originality_subject_aesthetics",
        "vhelm_image_critique_subject_aesthetics",
    ],
    "originality": ["vhelm_image_critique_originality_subject_aesthetics"],
    "photorealism": ["vhelm_photorealism"],
}

QUESTION_TYPE_TO_ANSWER_KEY: Dict[str, str] = {
    "alignment": "Answer.image_text_alignment_human.{choice}.on",
    "clear_subject": "Answer.clear_subject_human.{choice}.on",
    "aesthetics": "Answer.aesthetics_human.{choice}.on",
    "originality": "Answer.originality_human.{choice}.on",
    "photorealism": "Answer.photorealism_human.{choice}.on",
}


def generate_heim_human_eval_dataset(raw_human_eval_results_path: str):
    """
    Given a human eval results folder from HEIM, generates a dataset that can be used to evaluate VLMs.

    vhelm_image_critique: reasoning and knowledge scenarios
        - alignment
    vhelm_image_critique_aesthetics: MSCOCO perturbations
        - alignment
        - aesthetics
    vhelm_image_critique_originality_subject_aesthetics: originality scenarios
        - alignment
        - originality
        - clear subject
        - aesthetics
    vhelm_image_critique_subject_aesthetics: MSCOCO, MSCOCO art styles, alignment scenarios
        - alignment
        - clear subject
        - aesthetics
    vhelm_photorealism: MSCOCO, MSCOCO perturbations
        - photorealism

    See https://docs.google.com/spreadsheets/d/1hAffl_eyBP7460Vf54WyOpfI5zVk3BfYb1sGGT10_Go/edit#gid=2108816474
    for reference.

    heim_human_eval/
        images/
            org1/
            org2/
            ...
        questions.json
        alignment.jsonl
        aesthetics.jsonl
        originality.jsonl
        clear_subject.jsonl
        photorealism.jsonl
    """

    def write_out_examples_to_jsonl(final_examples: List[Dict[str, Any]], examples_path: str):
        with open(examples_path, "w") as examples_file:
            for final_example in final_examples:
                examples_file.write(json.dumps(final_example) + "\n")

    output_path: str = "heim_human_eval"
    if os.path.exists(output_path):
        # Delete the folder if it already exists
        shutil.rmtree(output_path)
        hlog("Deleted existing output folder.")
    os.makedirs(output_path)
    hlog(f"Created output folder at {output_path}")
    images_folder: str = "images"
    images_path: str = os.path.join(output_path, images_folder)
    os.makedirs(images_path)

    # Write out the questions to questions.json
    questions_path: str = os.path.join(output_path, "questions.json")
    with open(questions_path, "w") as f:
        f.write(json.dumps(QUESTION_TYPE_TO_INFOS, indent=3))
    hlog(f"Wrote out questions to {questions_path}.\n")

    for question_type, question_info in QUESTION_TYPE_TO_INFOS.items():
        with htrack_block(f"Processing question type {question_type}"):

            # Keep track of the examples for this question type. Use the image url as the key
            examples: Dict[str, Dict[str, Any]] = {}

            question_folders: List[str] = QUESTION_TYPES_TO_FOLDERS[question_type]
            hlog(f"Processing {len(question_folders)} question folders: {', '.join(question_folders)}...\n")

            for question_folder in question_folders:
                question_folder_path: str = os.path.join(raw_human_eval_results_path, question_folder)
                if not os.path.exists(question_folder_path):
                    raise ValueError(f"Question folder {question_folder_path} does not exist.")

                with htrack_block(f"Processing question folder {question_folder}"):
                    # Read the CSV files in the folder
                    for csv_file_name in os.listdir(question_folder_path):
                        if not csv_file_name.startswith("Batch"):
                            continue

                        with htrack_block(f"Processing CSV file {csv_file_name}"):
                            csv_file_path: str = os.path.join(question_folder_path, csv_file_name)
                            with open(csv_file_path, "r") as csv_file:
                                reader = csv.DictReader(csv_file)
                                for row in tqdm(reader):
                                    image_url: str = row["Input.image"]
                                    image_org: str = "mscoco" if "mscoco" in image_url else image_url.split("/")[-2]
                                    image_file_name: str = image_url.split("/")[-1]
                                    org_folder_path: str = os.path.join(images_path, image_org)
                                    os.makedirs(org_folder_path, exist_ok=True)
                                    local_image_path: str = os.path.join(org_folder_path, image_file_name)
                                    download_image_if_not_exists(image_url, local_image_path)

                                    choices = question_info["choices"]
                                    correct_answer: int = -1
                                    for choice in choices:
                                        answer_key: str = QUESTION_TYPE_TO_ANSWER_KEY[question_type].format(
                                            choice=choice
                                        )
                                        answer = row[answer_key]
                                        if answer == "true":
                                            correct_answer = choice  # type: ignore
                                            break
                                    assert int(correct_answer) != -1, f"Could not find correct answer for {image_url}."

                                    relative_image_path: str = os.path.join(images_folder, image_org, image_file_name)
                                    if image_url not in examples:
                                        examples[image_url] = {
                                            "image_path": relative_image_path,
                                            "human_annotations": [correct_answer],
                                        }
                                        if "Input.prompt" in row:
                                            examples[image_url]["prompt"] = row["Input.prompt"]
                                    else:
                                        examples[image_url]["human_annotations"].append(correct_answer)

            # Shuffle examples
            shuffled_examples = list(examples.values())
            random.shuffle(shuffled_examples)

            # Compute the mean score for each example
            for example in shuffled_examples:
                assert len(example["human_annotations"]) > 5, f"Expected 5 or 10 human annotations for {example}"
                example["mean_score"] = statistics.fmean(example["human_annotations"])

                # Check that the image exists and it has some pixels
                local_image_path = os.path.join(output_path, example["image_path"])
                assert (
                    os.path.exists(local_image_path) and os.path.getsize(local_image_path) > 0
                ), f"Image {local_image_path} does not exist."

            # Create train, valid, and test splits (80/10/10) ratio
            num_examples: int = len(shuffled_examples)
            num_train_examples: int = int(0.8 * num_examples)
            num_valid_examples: int = int(0.1 * num_examples)
            num_test_examples: int = num_examples - num_train_examples - num_valid_examples
            assert num_train_examples + num_valid_examples + num_test_examples == num_examples

            train_examples: List[Dict[str, Any]] = shuffled_examples[:num_train_examples]
            valid_examples: List[Dict[str, Any]] = shuffled_examples[
                num_train_examples : num_train_examples + num_valid_examples
            ]
            test_examples: List[Dict[str, Any]] = shuffled_examples[num_train_examples + num_valid_examples :]

            # Write out the examples to JSONL files
            train_examples_path: str = os.path.join(output_path, f"{question_type}_train.jsonl")
            write_out_examples_to_jsonl(train_examples, train_examples_path)
            valid_examples_path: str = os.path.join(output_path, f"{question_type}_valid.jsonl")
            write_out_examples_to_jsonl(valid_examples, valid_examples_path)
            test_examples_path: str = os.path.join(output_path, f"{question_type}_test.jsonl")
            write_out_examples_to_jsonl(test_examples, test_examples_path)

            # Print stats
            hlog(f"\n\nNumber of train examples: {len(train_examples)}")
            hlog(f"Number of valid examples: {len(valid_examples)}")
            hlog(f"Number of test examples: {len(test_examples)}")
            hlog(f"Total number of examples: {len(shuffled_examples)}\n\n")


def download_image_if_not_exists(image_url: str, local_image_path: str) -> None:
    if os.path.exists(local_image_path):
        return

    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError(f"Image URL {image_url} returned status code {response.status_code}.")
    with open(local_image_path, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("human_eval_results_path", type=str, help="Path to human eval results.")
    args = parser.parse_args()
    generate_heim_human_eval_dataset(args.human_eval_results_path)
