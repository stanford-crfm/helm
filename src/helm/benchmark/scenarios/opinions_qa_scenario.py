import os
import pandas as pd
from typing import List, Dict

from helm.common.general import ensure_file_downloaded

from helm.benchmark.scenarios.scenario import (
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
    2. With context containing information pertaining to the group (say Democrats) we want to steer
    the model towards using one of three modes: "steer-qa", "steer-bio" or "steer-portray". See the
    example below for an illustration fo the difference between the three steering modes.


    We prompt models using the following format

        <optional context>

        Question: <question>
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        Answer: <A/B/C/D>

    For instance:

        <optional context>

        Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes
        to gun violence in the country today?
        A. A great deal
        B. A fair amount
        C. Not too much
        D. Not at all
        E. Refused
        Answer:


    In the example above, the <optional context> varies based on the choice of the "context" variable as follows:

    1. If context="default", <optional context> is not used and the model is prompted with the question directly.

    2. If context="steer-qa", the group information (to steer the model towards) is passed in a QA format, e.g.:

        Question: In politics today, do you consider yourself a
        A. Republican
        B. Democrat
        C. Independent
        D. Something else
        E. Refused
        Answer: B

    3. If context="steer-bio", the group information is passed to the model as a free-text response
    to a prompt asking for demographic information, e.g.:

    Below you will be asked to provide a short description of your political affiliation and
    then answer some questions.

    Description: In politics today, I consider myself a Democrat.

    4. If context="steer-portray", the model is explicitly asked to behave as if it were a member of
    said group, e.g.:

    Answer the following question as if in politics today, you considered yourself a Democrat.

    """

    name = "opinions_qa"
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

    def download_data(self, output_path: str):
        data_dir: str = os.path.join(output_path, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        DOWNLOAD_FILENAMES = [self.FILE_NAME.format(wave=wave) for wave in self.PEW_SURVEY_WAVES]
        DOWNLOAD_FILENAMES += [f"{steer}.csv" for steer in ["steer-qa", "steer-bio", "steer-portray"]]
        DOWNLOAD_FILENAMES += ["Pew_American_Trends_Panel_disagreement_500.csv"]

        for filename in DOWNLOAD_FILENAMES:
            data_path: str = os.path.join(data_dir, filename)

            source_url: str = self.CODALAB_URI_TEMPLATE.format(bundle=self.CODALAB_BUNDLE, filename=filename)
            ensure_file_downloaded(source_url=source_url, target_path=data_path, downloader_executable="gdown")

    def read_survey_questions(self, csv_path):
        df = pd.read_csv(csv_path, sep="\t")
        df["options"] = df.apply(lambda x: eval(x["options"]), axis=1)
        return df

    def get_instances(self, output_path: str) -> List[Instance]:
        self.download_data(output_path)

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "dev": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }

        all_splits = ["dev", "test"] if self.context == "steer-qa" else ["test"]
        csv_dict = {
            "dev": os.path.join(output_path, f"{self.context}.csv"),
            "test": os.path.join(output_path, f"{self.survey_type}.csv"),
        }

        bios_df = None
        if self.context in ["steer-bio", "steer-portray"]:
            bios_path = os.path.join(output_path, f"{self.context}.csv")
            bios_df = pd.read_csv(bios_path, sep="\t")

        for split in all_splits:
            csv_path: str = csv_dict[split]
            assert os.path.exists(csv_path)

            question_df = self.read_survey_questions(csv_path)

            for qidx, (question, answers) in enumerate(zip(question_df["question"], question_df["options"])):
                # Opinions QA test questions have no correct answer and thus we set it to be None by default
                # for all test instances.
                # In the case where context = steer-qa, we add demographic information in the form of a
                # in-context question answer pair as shown in the example above.

                correct_answer = None if split == "test" else question_df["correct"][qidx]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(
                        Output(text=answer),
                        tags=[CORRECT_TAG] if (answer == correct_answer and split != "test") else [],
                    )

                if bios_df is None:
                    # context = "default" or "steer-qa"
                    instance = Instance(
                        Input(text=question),
                        references=list(map(answer_to_reference, answers)),
                        split=splits[split],
                    )
                    instances.append(instance)
                else:
                    # context = "steer-bio"or "steer-portray"
                    for bio in bios_df["question"].values:
                        context = PassageQuestionInput(passage=bio, question=question + "\n")
                        instance = Instance(
                            context,
                            references=list(map(answer_to_reference, answers)),
                            split=splits[split],
                        )
                        instances.append(instance)

        return instances
