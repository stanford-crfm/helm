import os
import pandas as pd
from typing import List, Dict

from helm.common.general import shell
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Input,
    Output,
)


class OpinionsQAScenario(Scenario):
    """
    The OpinionsQAScenario dataset is from the paper "Whose Opinions Do Language Models Reflect?"
    [Santurkar et al., 2023].

    OpinionsQA is a QA dataset containing 1484 multiple-choice questions. Since the questions are inherently
    subjective, there isn't a single ground truth response. Instead, the object of interest is how
    the distribution of model responses compares to those obtained from human survey participants.

    As discussed in Santurkar et al., we consider prompting an LM:
    1. Without any context (zero-shot) to evaluate the "default" opinions reflected
        by it.
    2. With context containing information pertaining to the group we want to steer the model towards.
       This context is either formatted as a question-answer pair (QA) or a textual description (BIO/PORTRAY).


    We prompt models using the following format

        <optional context>

        Question: <question>
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        Answer: <A/B/C/D>

    For example, we have:

        Question: In politics today, do you consider yourself a
        A. Republican
        B. Democrat
        C. Independent
        D. Something else E. Refused
        Answer: B

        Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes
        to gun violence in the country today?
        A. A great deal
        B. A fair amount
        C. Not too much
        D. Not at all
        E. Refused
        Answer:


    """

    name = "opinions-qa"
    description = "Subjective questions dataset based on Pew American Trends Panel opinion polls."
    tags = ["multiple_choice"]

    """ Information needed to download the dataset """
    CODALAB_URI_TEMPLATE: str = (
        "https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/model_input/{filename}"
    )
    CODALAB_BUNDLE: str = "0xa6f81cc62d7d4ccb93031a72d2043669"
    FILE_NAME: str = "Pew_American_Trends_Panel_W{wave}.csv"
    PEW_SURVEY_WAVES: list = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]

    def __init__(self, survey_type: str, context: str):
        super().__init__()
        assert context in ["default", "steer-qa", "steer-bio", "steer-portray"]

        self.survey_type: str = survey_type
        self.context: str = context

    def download_data(self):

        DOWNLOAD_FILENAMES = [self.FILE_NAME.format(wave=wave) for wave in self.PEW_SURVEY_WAVES]
        DOWNLOAD_FILENAMES += [f"{steer}.csv" for steer in ["steer-qa", "steer-bio", "steer-portray"]]
        DOWNLOAD_FILENAMES += ["Pew_American_Trends_Panel_disagreement_500.csv"]

        for filename in DOWNLOAD_FILENAMES:
            data_path: str = os.path.join(self.output_path, filename)

            source_url: str = self.CODALAB_URI_TEMPLATE.format(bundle=self.CODALAB_BUNDLE, filename=filename)
            if not os.path.exists(data_path):
                shell(["wget", source_url, "--no-check-certificate", "-O", data_path])

    def read_survey_questions(self, csv_path):
        df = pd.read_csv(csv_path, sep="\t")
        df["options"] = df.apply(lambda x: eval(x["options"]), axis=1)
        return df

    def get_instances(self) -> List[Instance]:

        self.output_path: str = os.path.join(self.output_path, "data")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.download_data()

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "dev": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }

        all_splits = ["dev", "test"] if self.context == "steer-qa" else ["test"]
        csv_dict = {
            "dev": os.path.join(self.output_path, f"{self.context}.csv"),
            "test": os.path.join(self.output_path, f"{self.survey_type}.csv"),
        }

        bios_df = None
        if self.context in ["steer-bio", "steer-portray"]:
            bios_path = os.path.join(self.output_path, f"{self.context}.csv")
            bios_df = pd.read_csv(bios_path, sep="\t")

        for split in all_splits:

            csv_path: str = csv_dict[split]
            assert os.path.exists(csv_path)

            question_df = self.read_survey_questions(csv_path)

            for qidx, (question, answers) in enumerate(zip(question_df["question"], question_df["options"])):

                # Opinions QA test questions have no correct answer. However, since the HELM codebase requires a
                # correct answer to be associated with each instance, we set it to be the first reference.
                # Note that this is never used in the analysis.
                # In the case where context = steer-qa, we add demographic information in the form of a
                # question answer pair as shown in the example above.

                correct_answer = answers[0] if split == "test" else question_df["correct"][qidx]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

                if bios_df is None:
                    # context = "default"/"steer-qa"
                    instance = Instance(
                        Input(text=question),
                        references=list(map(answer_to_reference, answers)),
                        split=splits[split],
                    )
                    instances.append(instance)
                else:
                    # context = "steer-bio"/"steer-portray"
                    for bio in bios_df["question"].values:

                        context = PassageQuestionInput(passage=bio, question=question + "\n")
                        instance = Instance(
                            context,
                            references=list(map(answer_to_reference, answers)),
                            split=splits[split],
                        )
                        instances.append(instance)

        return instances
