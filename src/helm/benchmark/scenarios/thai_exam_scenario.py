import os
from typing import Dict, List
import json

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class ThaiExamScenario(Scenario):
    """
    ThaiExam, a benchmark comprising Thai multiple-choice examinations as follows:

    ∙ ONET: The Ordinary National Educational Test (ONET) is an examination for students in Thailand.
    We select the grade-12 ONET exam, which comprises 5 subjects and each question has 5 choices.
    These subjects are Thai, English, Mathematics, Social Studies, and Science.
    Amounting to a total of 170 questions and options.

    ∙ IC: The Investment Consultant (IC) examination, a licensing test for investment professionals in Thailand.
    Developed by the Stock Exchange of Thailand (SET), features 4 choices per question.
    We extracted questions for levels 1, 2, and 3 resulting in a total of 95 questions and options.

    ∙ TGAT: The Thai General Aptitude Test (TGAT), a national high school examination in Thailand.
    Focuses on critical and logical thinking skills.
    We collected a total of 90 questions and answers. The TGAT consists of four choices per question.

    ∙ TPAT-1: The Thai Professional Aptitude Test 1 (TPAT-1) is a national high school examination in Thailand.
    The Exam assesses students’ professional skills requirement in medical schools.
    This subset contains reasoning and medical ethics. We collected a total of 116 questions and answers.
    The TPAT-1 consists of 5 choices per question.

    ∙ A-Level: An academic knowledge assessment examination (Applied Knowledge Level)
    that covers general foundational subjects taught in schools.
    The content assessed in this examination aligns with the curriculum guidelines
    and emphasizes the practical application of knowledge in daily life.
    We collected a total of 175 questions and answers.

    We created and used these exams to evaluate the performance of the Typhoon models(https://arxiv.org/abs/2312.13951).

    Prompt models using the following format

        <input>                  # train
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        E. <reference>
        Answer: <A/B/C/D/E>

        x N (N-shot)

        <input>                  # test
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        E. <reference5>
        Answer:

    For example:

        ในระบบย่อยอาหารของมนุษย์ การดูดซึมสารอาหารส่วนใหญ่เกิดขึ้นที่อวัยวะใด?
        A. ลำไส้เล็ก
        B. ตับอ่อน
        C. ลำไส้ใหญ่
        D. กระเพาะอาหาร
        E. หัวใจ
        Answer: A

        ข้อใดอธิบายเกี่ยวกับแรงไฟฟ้าได้ถูกต้อง?
        A. เกิดได้โดยที่วัตถุไม่ต้องสัมผัสกัน
        B. เป็นได้เฉพาะแรงผลักเท่านั้น
        C. เป็นได้เฉพาะแรงดูดเท่านั้น
        D. เป็นแรงต้านระหว่างวัตถุเท่านั้น
        E. ถูกทุกข้อ
        Answer:

    Target: A
    """

    name = "thai_exam"
    description = "ThaiExam benchmark comprising Thai multiple-choice examinations."
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, exam: str):
        super().__init__()
        self.exam = exam

    def download_thai_exam(self, path: str, revision: str):
        ensure_file_downloaded(
            f"https://huggingface.co/datasets/scb10x/thai_exam/resolve/{revision}/thai_exam.tar.gz",
            target_path=path,
            unpack=True,
        )

    def process_jsonl(self, jsonl_path: str, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Reading {jsonl_path}")
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                # for handle missing key incase of some subject doesn't have all 5 choices
                answers = [data[key] for key in ["a", "b", "c", "d", "e"] if key in data and data[key] != ""]
                answers_dict = dict(zip(["A", "B", "C", "D", "E"], answers))

                question, correct_answer = data["question"], answers_dict[data["answer"].upper()]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=Input(text=question),
                    references=list(map(answer_to_reference, answers)),
                    split=split,
                )
                instances.append(instance)
        return instances

    def get_instances(self, output_path) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        # ThaiExam (v1.0) revision = d78aef04ea3cc5095545e6951cb39e17c64e26a1
        self.download_thai_exam(data_path, revision="d78aef04ea3cc5095545e6951cb39e17c64e26a1")
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "train": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
        for split in splits:
            jsonl_path: str = os.path.join(data_path, self.exam, f"{self.exam}_{split}.jsonl")
            if not os.path.exists(jsonl_path):
                hlog(f"{jsonl_path} doesn't exist, skipping")
                continue
            instances.extend(self.process_jsonl(jsonl_path, splits[split]))

        return instances
