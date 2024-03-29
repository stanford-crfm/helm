import csv
import os
from typing import Dict, List
import json

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class ThaiExamScenario(Scenario):
    """
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
    description = "Thai Exam dataset from ONET, TGAT, TPAT1, A-Level"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, exam: str):
        super().__init__()
        self.exam = exam

    def download_thai_exam(self, path: str):
        ensure_file_downloaded(
            "https://storage.googleapis.com/helm-benchmark/thai_exam.tar.gz",
            target_path=path,
            unpack=True,
        )
    
    def process_jsonl(self, jsonl_path: str, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Reading {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # for handle missing key incase of some subject doesn't have all 5 choices
                answers = [data.get(key, None) for key in ["a", "b", "c", "d", "e"] if data.get(key, None) != ""] 
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
        # Download the raw data
        data_path: str = os.path.join(output_path, "data")
        self.download_thai_exam(data_path)

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {            
            "train": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
        for split in splits:
            jsonl_path: str = os.path.join(data_path, {self.exam}, f"{self.exam}_{split}.jsonl")
            if not os.path.exists(jsonl_path):
                hlog(f"{jsonl_path} doesn't exist, skipping")
                continue
            instances.extend(self.process_jsonl(jsonl_path, splits[split]))

        return instances