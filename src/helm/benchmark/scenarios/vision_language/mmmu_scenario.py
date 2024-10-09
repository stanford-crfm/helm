import os.path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class MMMUScenario(Scenario):
    """
    MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI

    We introduce MMMU: a new benchmark designed to evaluate multimodal models on massive multi-discipline
    tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously
    collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines:
    Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering.
    These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such
    as charts, diagrams, maps, tables, music sheets, and chemical structures.

      @article{yue2023mmmu,
        title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
        author={Xiang Yue and Yuansheng Ni and Kai Zhang and Tianyu Zheng and Ruoqi Liu and Ge Zhang and Samuel
        Stevens and Dongfu Jiang and Weiming Ren and Yuxuan Sun and Cong Wei and Botao Yu and Ruibin Yuan and
        Renliang Sun and Ming Yin and Boyuan Zheng and Zhenzhu Yang and Yibo Liu and Wenhao Huang and Huan Sun
        and Yu Su and Wenhu Chen},
        journal={arXiv preprint arXiv:2311.16502},
        year={2023},
      }

    Paper: https://arxiv.org/abs/2311.16502
    Website: https://mmmu-benchmark.github.io/
    """

    MMMU_HUGGINGFACE_DATASET_NAME: str = "MMMU/MMMU"
    MAX_NUM_IMAGES: int = 7

    SUBJECTS: List[str] = [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ]

    name = "mmmu"
    description = (
        "Evaluate multimodal models on massive multi-discipline tasks demanding college-level "
        "subject knowledge and deliberate reasoning ([Yue et al., 2023](https://arxiv.org/abs/2311.16502))."
    )
    tags = ["vision-language"]

    def __init__(self, subject: str, question_type: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

        assert question_type in ["multiple-choice", "open"], f"Invalid question type: {question_type}"
        self._question_type: str = question_type

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images", self._subject)
        ensure_directory_exists(images_path)

        instances: List[Instance] = []

        # Process the validation set
        # There seems to be a dev set, but it's unavailable through load_dataset.
        # When loading "dev", we get error: ValueError: Unknown split "dev". Should be one of ['test', 'validation'].
        # The test set doesn't have answers, since the MMMU competition/leaderboard uses the test set
        for row in tqdm(
            load_dataset(self.MMMU_HUGGINGFACE_DATASET_NAME, self._subject, split="validation", cache_dir=output_path)
        ):
            # Skip questions that aren't in the subject we're interested in
            if row["question_type"] != self._question_type:
                continue

            question_id: str = row["id"]
            if self._subject not in question_id:
                continue

            question_template: str = row["question"]
            options: List[str] = eval(row["options"])  # Convert the string to list of options
            answer: str = row["answer"]

            # Create the question. Questions can have text and images interleaved
            question_template_to_image_path: Dict[str, str] = {}
            content: List[MediaObject] = []
            for img_number in range(1, self.MAX_NUM_IMAGES):
                image_id: str = f"image_{img_number}"
                if row[image_id] is None:
                    # At this point, there are no more images for this question
                    break

                # Save the image locally
                image_path: str = os.path.join(images_path, f"{question_id}_{image_id}.png")
                if not os.path.exists(image_path):
                    row[image_id].save(image_path)

                image_template_tag: str = f"<image {img_number}>"
                question_template_to_image_path[image_template_tag] = image_path

                # There are cases when the image is included, but it is not used either in the
                # question template or in the answer options
                if image_template_tag not in question_template:
                    # The image is not in the question template
                    continue

                head, question_template = question_template.split(image_template_tag, 1)
                if head:
                    content.append(MediaObject(text=head, content_type="text/plain"))
                content.append(MediaObject(location=image_path, content_type="image/png"))

            # Add the rest of the question template
            if question_template:
                content.append(MediaObject(text=question_template, content_type="text/plain"))

            # Add the references
            references: List[Reference] = []
            if self._question_type == "multiple-choice":
                # The given correct answer is a letter, but we need an index
                correct_answer_index: int = ord(answer) - ord("A")

                for i, option in enumerate(options):
                    reference: Reference
                    is_correct: bool = i == correct_answer_index

                    if option in question_template_to_image_path:
                        # Image answer choices
                        image_path = question_template_to_image_path[option]
                        image_choice = MediaObject(location=image_path, content_type="image/png")
                        reference = Reference(
                            Output(multimedia_content=MultimediaObject([image_choice])),
                            tags=[CORRECT_TAG] if is_correct else [],
                        )
                    else:
                        # Text answer choices
                        reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])

                    references.append(reference)
            else:
                references.append(Reference(Output(text=answer), tags=[CORRECT_TAG]))

            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=references,
                    split=VALID_SPLIT,
                )
            )

        return instances
