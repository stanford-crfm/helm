import json
import os
import string
from typing import Dict, List
import pandas as pd
from typing import List, Dict, Optional

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
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


class LMOpinionsScenario(Scenario):
    """
    The LMOpinions dataset is from the paper:
    [TODO ARXIV]


    LM Opinions is a QA dataset containing 1484 multiple-choice questions. Since the questions are inherently 
    subjective, there isn't a single ground truth response. Instead, the object of interest is how 
    the distribution of model responses compares to those obtained from human survey participants as 
    disccused in [TODO GITHUB].
    
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

    name = "lm-opinions"
    description = "Subjective questions dataset based on Pew American Trends Panel opinion polls."
    tags = ["multiple_choice"]

    def __init__(self, survey_type: str, context: str):
        super().__init__()
        assert context in ['default', "steer-qa", "steer-bio", "steer-portray"]
        
        self.survey_type: str = survey_type
        self.context: str = context
            
    def read_survey_questions(self, csv_path):
        df = pd.read_csv(csv_path, sep="\t")
        df["options"] = df.apply(lambda x: eval(x["options"]), axis=1)
        return df
        

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(self.output_path, "data")
        # ensure_file_downloaded(
        #    source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        #    target_path=data_path,
        #    unpack=True,
        #    unpack_type="untar",
        # )

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "dev": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
            

        all_splits = ["dev", "test"] if self.context == "steer-qa" else ["test"]
        csv_dict = {"dev": os.path.join(data_path, f"{self.context}.csv"),
                    "test": os.path.join(data_path, f"{self.survey_type}.csv")}
        
        bios_df = None
        if self.context in ["steer-bio", "steer-portray"]:
            bios_path = os.path.join(data_path, f"{self.context}.csv")
            bios_df = pd.read_csv(bios_path, sep="\t")     
                    
        for split in all_splits:

            csv_path: str = csv_dict[split]
            assert os.path.exists(csv_path)
            
            question_df = self.read_survey_questions(csv_path)

            for qidx, (question, answers) in enumerate(zip(question_df["question"], 
                                                           question_df["options"])):

                prefixes = list(string.ascii_uppercase)
                assert len(prefixes) >= len(answers)
                answers_dict = dict(zip(prefixes, answers))
                
                # LM Opinions test questions have no correct answer. However, since the HELM codebase requires a
                # correct answer to be associated with each instance, we set it to be the first reference.
                # Note that this is never used in the analysis.
                # In the case where context = steer-qa, we add demographic information in the form of a
                # question answer pair as shown in the example above.
                
                correct_answer = answers[0] if split == "test" else question_df["correct"][qidx]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

                if bios_df is None:
                    # context = "default"/"steer-qa"
                    instance = Instance(Input(text=question),
                        references=list(map(answer_to_reference, answers)),
                        split=splits[split],
                    )
                else:
                    # context = "steer-bio"/"steer-portray"
                    for bio in bios_df['question'].values:

                        context = PassageQuestionInput(passage=bio, question=question+'\n')
                        instance = Instance(context,
                            references=list(map(answer_to_reference, answers)),
                            split=splits[split],
                        )
                instances.append(instance)

        return instances