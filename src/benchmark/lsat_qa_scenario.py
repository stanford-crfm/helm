import os
import json
from typing import List

from common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


class LSATScenario(Scenario):
    """
    The LSAT dataset is from the paper:
        https://arxiv.org/abs/2104.06598

    Original repository can be found at:
        https://github.com/facebookarchive/bAbI-tasks

    This is a multi-choice QA dataset containing question that test analytical reasoning,
    from the Law School Admission Test (LSAT).

    We prompt models using the following format:
        Passage: <passage>
        Question: <question>
        A. ...
        B. ...
        C. ...

        Target completion:
            <answer>

    Using an example from the training dataset, we have
            Passage: Of the eight students - George, Helen, Irving, Kyle, Lenore, Nina, Olivia, and Robert -
            in a seminar, exactly six will give individual oral reports during three consecutive days - Monday,
            Tuesday, and Wednesday. Exactly two reports will be given each day - one in the morning and one in
            the afternoon - according to the following conditions: Tuesday is the only day on which George can
            give a report. Neither Olivia nor Robert can give an afternoon report. If Nina gives a report, then
            on the next day Helen and Irving must both give reports, unless Nina's report is given on Wednesday.

            Question: Which one of the following could be the schedule of the students' reports?
            A. Mon. morning: Helen; Mon. afternoon: Robert Tues. morning: Olivia; Tues. afternoon: Irving Wed.
               morning: Lenore; Wed. afternoon: Kyle
            B. Mon. morning: Irving; Mon. afternoon: Olivia Tues. morning: Helen; Tues. afternoon: Kyle Wed.
               morning: Nina; Wed. afternoon: Lenore
            C. Mon. morning: Lenore; Mon. afternoon: Helen Tues. morning: George; Tues. afternoon: Kyle Wed.
               morning: Robert; Wed. afternoon: Irving
            D. Mon. morning: Nina; Mon. afternoon: Helen Tues. morning: Robert; Tues. afternoon: Irving Wed.
               morning: Olivia; Wed. afternoon: Lenore
            E. Mon. morning: Olivia; Mon. afternoon: Nina Tues. morning: Irving; Tues. afternoon: Helen Wed.

        Target completion:
            C
    """

    name = "lsat_qa"
    description = "Question answering dataset with questions from LSAT exams."
    tags = ["question_answering"]

    def __init__(self, task):
        self.task = task
        question_types = {
            "grouping": ["in/out grouping", "distribution grouping"],
            "ordering": ["simple ordering", "relative ordering", "complex ordering"],
            "assignment": ["determined assignment", "undetermined assignment"],
            "miscellaneous": [],
        }
        
        self.subtype2type = {}
        for qtype, subtypes in question_types.items():
            for subtype in subtypes:
                self.subtype2type[subtype] = qtype

    def get_question_types(self, tags):
        qtype: str = tags[2].replace("grouping (distribution)", "distribution grouping") or "miscellaneous"
        return [qtype.replace(" ", "_").replace("/", "_"), self.subtype2type.get(qtype)]

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        splits = {"train": ("Training", TRAIN_TAG), "valid": ("Development", VALID_TAG), "test": ("Test", TEST_TAG)}

        train_num: int = 1000
        for split in splits:
            split_name = splits[split][0]
            url = f"https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_{split_name}Data.json"
            target_path = f"{data_path}/AR_{split_name}Data.json"
            ensure_file_downloaded(source_url=url, target_path=target_path)

            with open(target_path, "r") as f:
                data = json.load(f)
                for p in data:
                    passage = p["passage"]
                    for q in p["questions"]:
                        qtypes = self.get_question_types(q["tags"])
                        if self.task == "all" or self.task in qtypes:
                            question = q["question"]
                            options = q["options"]
                            answer = ord(q["answer"]) - ord("A")
                            context = f"Passage: {passage}\nQuestion: {question}"
                            
                            references: List[Reference] = []
                            for index, option in enumerate(options):
                                tags = [CORRECT_TAG] if index == answer else []
                                references.append(Reference(output=option, tags=tags))

                            tags = [TRAIN_TAG, VALID_TAG, TEST_TAG]
                            instance: Instance = Instance(input=context, references=references, tags=tags)
                            instances.append(instance)

        return instances
