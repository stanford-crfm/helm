import os
import json
from typing import Dict, List

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)


class LSATScenario(Scenario):
    """
    The LSAT dataset is from the paper:
    https://arxiv.org/abs/2104.06598

    Original repository can be found at:
    https://github.com/zhongwanjun/AR-LSAT

    This is a multi-choice QA dataset containing question that test analytical reasoning,
    from the Law School Admission Test (LSAT). The questions explore cases of constraint
    satisfaction, where there is a set of elements that need to be assigned while complying
    with given conditions, for instance: making 1-1 assignments of talks to dates ("assignment"),
    grouping students to teams ("grouping") or ordering classes in a schedule ("ordering").

    We can either evaluate all questions together ("all") or a subset of the questions:

    - grouping: in_out_grouping, distribution_grouping
    - ordering: simple ordering, relative_ordering, complex ordering
    - assignment: determined assignment, undetermined assignment
    - miscellaneous

    We prompt models using the following format:

    Input

        Passage: <passage>
        Question: <question>
        A. ...
        B. ...
        C. ...

    Output (Target completion)

        B

    Using an example from the training dataset, we have:

    Input

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

    Target completion

        C
    """

    name = "lsat_qa"
    description = "Question answering dataset with questions from LSAT exams."
    tags = ["question_answering"]

    def __init__(self, task):
        super().__init__()
        question_types: Dict[str, List[str]] = {
            "grouping": ["in/out grouping", "distribution grouping"],
            "ordering": ["simple ordering", "relative ordering", "complex ordering"],
            "assignment": ["determined assignment", "undetermined assignment"],
            "miscellaneous": [],
        }

        # Validate that task is one of the keys in `question_types` or "all"
        assert task in question_types or task == "all", f"Invalid task: {task}"
        self.task = task

        self.subtype2type = {}
        for question_type, subtypes in question_types.items():
            for subtype in subtypes:
                self.subtype2type[subtype] = question_type

    def get_question_types(self, tags: List[str]) -> List[str]:
        question_type: str = tags[2].replace("grouping (distribution)", "distribution grouping") or "miscellaneous"
        types = [question_type.replace(" ", "_").replace("/", "_")]
        main_type = self.subtype2type.get(question_type)
        if main_type is not None:
            types.append(main_type)
        return types

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        splits = {"Training": TRAIN_SPLIT, "Development": VALID_SPLIT, "Test": TEST_SPLIT}

        for split in splits:
            url = f"https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_{split}Data.json"
            target_path = f"{data_path}/AR_{split}Data.json"
            ensure_file_downloaded(source_url=url, target_path=target_path)

            with open(target_path, "r") as f:
                data = json.load(f)
                for p in data:
                    passage = p["passage"]
                    for q in p["questions"]:
                        question_types: List[str] = self.get_question_types(q["tags"])
                        if self.task == "all" or self.task in question_types:
                            question = q["question"]
                            options = q["options"]
                            answer = ord(q["answer"]) - ord("A")

                            references: List[Reference] = []
                            for index, option in enumerate(options):
                                tags = [CORRECT_TAG] if index == answer else []
                                references.append(Reference(Output(text=option), tags=tags))

                            instance: Instance = Instance(
                                input=PassageQuestionInput(passage=passage, question=question),
                                references=references,
                                split=splits[split],
                            )
                            instances.append(instance)

        return instances
