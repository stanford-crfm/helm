import os
import json
from random import Random
from typing import Any, List, Dict

import pandas as pd
from pandas import DataFrame

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
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


class CtiToMitreScenario(Scenario):
    """
    Original Task:
    - The original task is to classify the description of the situation regarding the system
      into the security threats in that situation.
    - The classification categories are the approximately 200 categories of attack techniques
      in the enterprise as defined by MITRE ATT&CK v10.1.

    Implemented Task:
    - Since classification into so many classes is difficult to handle in a generative language model
      such as GPT itself, we implement this task as a multiple-choice task.
    - Each choice is the name of the attack technique category into which the description is classified.
    - The number of options is determined by the parameter (num_options).
        - The minimum number of options is 2 and the maximum is 199, the number of all categories of
          attack methods defined in MITRE ATT&CK v10.1.
    - From the 199 choices, num_options choices, including the correct answer and a default case,
      are randomly selected and used.
        - If num_options is not specified, all 199 category names will be used as choices.

    Data:
    - dataset.csv
        - Target dataset
        - https://github.com/dessertlab/cti-to-mitre-with-nlp/raw/a8cacf3185d098c686e0d88768a619a03a4d76d1/data/dataset.csv
        - This data is of the form [sentence, label_tec, label_subtec, tec_name]
            - sentence: the description
            - label_tec: label for attack technique category
            - label_subtec: label for attack technique subcategory
            - tec_name : name(simple description) for attack technique subcategory
                - Note: we need to extract name for attack technique category
                        from enterprise-attack.json

    - enterprise-attack.json
        - https://github.com/mitre/cti/archive/refs/tags/ATT&CK-v10.1.zip
            - /mitre_v10/enterprise-attack/enterprise-attack.json
        - This data contains relation from attack technique name to attack technique label
            - we can extract attack technique category name for label_tec using this json data.


    Prompt: (k is specified by num_options)
        -----------------------
        Answer the possible security attacks in each of the following situations from each of the options below.
                                        [instruction]

        Situation: <description>        [in context examples]
        A. <attack_category_name_1>
        B. <attack_category_name_2>
        ...
        Y. <attack_category_name_(k-1)>
        Z. Others
        Answer: <correct_answer>

        ... (Examples are output as long as the length allows) ...

        Situation: <target_description>  [target question]
        A. <attack_category_name_t1>
        B. <attack_category_name_t2>
        ...
        Y. <attack_category_name_t(k-1)>
        Z. Others
        Answer:
        -----------------------

    Example of prompt (num_options = 5)
        -----------------------
        Answer the possible security attacks in each of the following situations from each of the options below.

        Situation: ZxShell can launch a reverse command shell.
        A. Command and Scripting Interpreter
        B. System Shutdown/Reboot
        C. Exfiltration Over C2 Channel
        D. Direct Volume Access
        E. Others
        Answer: A

        ....(Omitted)...

        Situation: APC injection is a method of executing arbitrary code in the address space.
        A. Event Triggered Execution
        B. Process Injection
        C. Non-Application Layer Protocol
        D. Escape to Host
        E. Others
        Answer: B

        Situation: Timestomping may be used along with file name Masquerading to hide malware and tools.
        A. Search Victim-Owned Websites
        B. Internal Spearphishing
        C. Application Layer Protocol
        D. Indicator Removal on Host
        E. Others
        Answer:
        -----------------------
    """  # noqa: E501

    # Names of the tasks we support
    name = "cti_to_mitre"
    description = "Classification of security attack opportunities on system"
    tags = ["classification", "cyber_security"]

    # Constant for splitting target data into train and test data.
    train_ratio = 0.7

    # Constant for default number of options. # of (MITRE ATT&CK attack categories) is 199 in ATT&CK-v10.1.zip
    MAX_NUM_OPTIONS = 199

    # Constant: the description for Others option
    OTHERS_OPTION = "Others"

    CTI_URL = "https://github.com/dessertlab/cti-to-mitre-with-nlp/raw/a8cacf3185d098c686e0d88768a619a03a4d76d1/data/dataset.csv"  # noqa: E501
    MITRE_URL = "https://github.com/mitre/cti/raw/refs/tags/ATT&CK-v10.1/enterprise-attack/enterprise-attack.json"

    def __init__(self, num_options: int = MAX_NUM_OPTIONS, seed: int = 42) -> None:
        """
        num_options: int, number of choices in multiple-choice task
        seed: int, seed for random module. The seed is set to random if specified
        """
        super().__init__()
        self.num_options = min(num_options, CtiToMitreScenario.MAX_NUM_OPTIONS)
        self.random_seed = seed
        self.random = Random(seed)

    @staticmethod
    def make_label_category_name_dict(jdata: Dict[str, Any]) -> Dict[str, str]:
        """
        This makes mapping from label_tec (attack technique category label) to tec_category_name
        (attack technique category name)
        - jdata is json object for enterprise_attack.json
        """

        category_id_to_name: Dict[str, str] = {}
        attacks = [
            o for o in jdata["objects"] if o["type"] == "attack-pattern" and not o.get("x_mitre_is_subtechnique", True)
        ]
        for attack in attacks:
            ids = [ref["external_id"] for ref in attack["external_references"] if ref["source_name"] == "mitre-attack"]
            assert len(ids) == 1
            id = ids[0]
            category_id_to_name[id] = attack["name"]
        return category_id_to_name

    def get_references(self, num_references: int, correct_cname: str, cnames: List[str]) -> List[Reference]:
        """
        Randomly select k tec_category_names (attack technique category names) as choices.
        However, choose not to include "excluded",
        and if k is less than the total number of possible choices, add a default case.
        - k : number of choices
        - correct_cname : correct attack technique category names
        - cnames : list containing all attack technique category names
        """
        assert num_references >= 2, "Need at least 2 references for the correct choice and 'Others'"
        num_incorrect_cname_samples = num_references - 2
        assert num_references <= len(
            cnames
        ), f"Cannot have more references than the number of categories, which is {len(cnames)}"
        incorrect_cnames = [cname for cname in cnames if cname != correct_cname]
        incorrect_cname_samples = self.random.sample(
            incorrect_cnames, min(len(incorrect_cnames), num_incorrect_cname_samples)
        )
        references = [Reference(Output(text=cname), tags=[]) for cname in incorrect_cname_samples]
        references.append(Reference(Output(text=correct_cname), tags=[CORRECT_TAG]))
        self.random.shuffle(references)
        if num_references <= len(cnames):
            references.append(Reference(Output(text=CtiToMitreScenario.OTHERS_OPTION), tags=[]))
        return references

    def create_multiple_choice_instances(
        self, df: DataFrame, split: str, label_cname: Dict[str, str]
    ) -> List[Instance]:
        """Create a list of instances corresponding to the multiple choice task"""
        instances = []
        for idx in df.index:
            linedata = df.loc[idx]
            sentence = linedata["sentence"]
            label_tec = linedata["label_tec"]
            correct_cname = label_cname[label_tec]
            all_cnames = [cname for cname in label_cname.values()]
            references = self.get_references(self.num_options, correct_cname, all_cnames)
            input = Input(text=sentence)
            instance = Instance(input, references, split=split)
            instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        dataset_path = os.path.join(data_dir, "dataset.csv")
        ensure_file_downloaded(
            source_url="https://github.com/dessertlab/cti-to-mitre-with-nlp/raw/a8cacf3185d098c686e0d88768a619a03a4d76d1/data/dataset.csv",  # noqa: E501
            target_path=dataset_path,
        )

        labels_path = os.path.join(data_dir, "enterprise-attack.json")
        ensure_file_downloaded(
            source_url="https://github.com/mitre/cti/raw/refs/tags/ATT&CK-v10.1/enterprise-attack/enterprise-attack.json",  # noqa: E501
            target_path=labels_path,
        )

        # load dataset
        all_df = pd.read_csv(dataset_path)

        # split all_df into train and test data frames
        train_df = all_df.sample(frac=CtiToMitreScenario.train_ratio, random_state=self.random_seed)
        test_df = all_df.drop(train_df.index).sample(frac=1, random_state=self.random_seed)

        # load labels
        with open(labels_path) as f:
            jdata = json.load(f)

        # make mapping from label_tec to tec_category_name
        label_cname = self.make_label_category_name_dict(jdata)

        # create instances from each dataset
        instances_train = self.create_multiple_choice_instances(train_df, TRAIN_SPLIT, label_cname)
        instances_test = self.create_multiple_choice_instances(test_df, TEST_SPLIT, label_cname)

        # return all instances
        all_instances = instances_train + instances_test
        return all_instances
