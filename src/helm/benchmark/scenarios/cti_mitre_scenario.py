import os
import json
import random
from typing import List, Dict
import pandas as pd
from pandas import DataFrame
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class CtiMitreScenario(Scenario):
    """
    Original Task:
    - The original task is to classify the description of the situation regarding the system
      into the security threats in that situation.
    - The classification categories are the approximately 200 categories of attack techniques
      in the enterprise as defined by MITRE ATT&CK v10.

    Implemented Task:
    - Since classification into so many classes is difficult to handle in a generative language model
      such as GPT itself, we implement this task as a multiple-choice task.
    - Each choice is the name of the attack technique category into which the description is classified.
    - The number of options is determined by the parameter (num_options).
        - The minimum number of options is 2 and the maximum is 199, the number of all categories of
          attack methods defined in MITRE ATT&CK v10.
    - From the 199 choices, num_options choices, including the correct answer and a default case,
      are randomly selected and used.
        - If num_options is not specified, all 199 category names will be used as choices.

    Data:
    - dataset.csv
        - Target dataset
        - https://github.com/dessertlab/cti-to-mitre-with-nlp/raw/main/data/dataset.csv
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
    """

    # Names of the tasks we support
    name = "cti_mitre"
    description = "Classification of security attack opportunities on system"
    tags = ["classification", "MITRE ATT&CK", "cyber_security"]

    # Constant for splitting target data into train and test data.
    train_ratio = 0.7

    # Constant for default number of options. # of (MITRE ATT&CK attack categories) is 199 in ATT&CK-v10.1.zip
    MAX_NUM_OPTIONS = 199

    # Constant: the description for Others option
    OTHERS_OPTION = "Others"

    # Methods

    def __init__(self, num_options=None, seed=None):
        """
        num_options: int, number of choices in multiple-choice task
        seed: int, seed for random module. The seed is set to random if specified
        """
        super().__init__()
        # dataset url
        self.dataset_all_url = "https://github.com/dessertlab/cti-to-mitre-with-nlp/raw/main/data/dataset.csv"
        self.dataset_all_name = "dataset.csv"
        # MITRE ATT CK (v10) url
        self.mitre_att_ck_v10_url = "https://github.com/mitre/cti/archive/refs/tags/ATT&CK-v10.1.zip"
        self.mitre_dir = "mitre_v10"
        self.enterprise_attack_dir = "enterprise-attack"
        self.enterprise_attack_json = "enterprise-attack.json"
        # Number of options : if num_options is not specified, num_options=MAX_NUM_OPTIONS
        if num_options is not None and 0 < num_options <= CtiMitreScenario.MAX_NUM_OPTIONS:
            self.num_options = num_options
        else:
            self.num_options = CtiMitreScenario.MAX_NUM_OPTIONS
        # set seed to random
        random.seed(seed)
        self.rand = random

    def download_dataset(self):
        """Download dataset.csv"""
        data_dir = self.data_dir
        ensure_directory_exists(data_dir)
        ensure_file_downloaded(
            source_url=self.dataset_all_url,
            target_path=os.path.join(data_dir, self.dataset_all_name),
        )

    def download_MITRE_info(self):
        """Download zip file containing enterprise_attack.json"""
        data_dir = self.data_dir
        ensure_directory_exists(data_dir)
        ensure_file_downloaded(
            source_url=self.mitre_att_ck_v10_url,
            target_path=os.path.join(data_dir, self.mitre_dir),
            unpack=True,
            unpack_type="unzip",
        )

    @staticmethod
    def make_label_category_name_dict(jdata) -> Dict[str, str]:
        """
        This makes mapping from label_tec (attack technique category label) to tec_category_name
        (attack technique category name)
        - jdata is json object for enterprise_attack.json
        """
        objs = jdata["objects"]
        label_cname: Dict[str, str] = {}
        if jdata is None:
            return label_cname
        for i in range(0, len(objs)):
            obj = objs[i]
            if obj["type"] == "attack-pattern":
                if "x_mitre_is_subtechnique" in obj and not obj["x_mitre_is_subtechnique"]:
                    extrefs = obj["external_references"]
                    label = None
                    for ref in extrefs:
                        if ref["source_name"] == "mitre-attack":  # and "external_id" in ref:
                            label = ref["external_id"]
                            break
                    if label is not None and "name" in obj:
                        cname = obj["name"]
                        label_cname[label] = cname
        return label_cname

    def select_option_cnames(self, k: int, excluded: str, cnames: List[str]) -> List[str]:
        """
        Randomly select k tec_category_names (attack technique category names) as choices.
        However, choose not to include "excluded",
        and if k is less than the total number of possible choices, add a default case.
        - k : number of choices
        - excluded : excluded attack technique category name (usually, specify correct answer)
        - cnames : list containing all attack technique category names
        """
        target_cnames = [v for v in cnames if v != excluded]

        if len(target_cnames) <= k:
            return target_cnames
        elif k - 1 <= 0:
            return [CtiMitreScenario.OTHERS_OPTION]
        else:
            ops = self.rand.sample(target_cnames, k - 1)
            ops.append(CtiMitreScenario.OTHERS_OPTION)
            return ops

    @staticmethod
    def bring_others_to_end(references: List[Reference]) -> List[Reference]:
        """Rearrange the list of references so that the reference corresponding to the default case comes last"""
        newref_list: List[Reference] = []
        others_list: List[Reference] = []
        for ref in references:
            if ref.output.text == CtiMitreScenario.OTHERS_OPTION:
                others_list.append(ref)
            else:
                newref_list.append(ref)
        newref_list.extend(others_list)
        return newref_list

    def create_multiple_choice_instances(
        self, df: DataFrame, split: str, label_cname: Dict[str, str]
    ) -> List[Instance]:
        """Create a list of instances corresponding to the multiple choice task"""
        instances = []
        for idx in df.index:
            linedata = df.loc[idx]
            sent = linedata["sentence"]
            label_tec = linedata["label_tec"]
            correct_cname = label_cname[label_tec]
            all_cnames = [cname for cname in label_cname.values()]
            num_of_wrong_options = self.num_options - 1
            wrong_cnames = self.select_option_cnames(num_of_wrong_options, correct_cname, all_cnames)
            input = Input(text=sent)
            # create options (including one correct answer)
            correct_ref = Reference(Output(text=correct_cname), tags=[CORRECT_TAG])
            references = [Reference(Output(text=cname), tags=[]) for cname in wrong_cnames]
            references.append(correct_ref)
            # shuffle answer options
            self.rand.shuffle(references)
            # bring others_option to the end of the reference list
            ord_references = CtiMitreScenario.bring_others_to_end(references)
            instance = Instance(input, ord_references, split=split)
            instances.append(instance)
        return instances

    def create_instances(self, df: DataFrame, split: str, label_cname: Dict[str, str]) -> List[Instance]:
        return self.create_multiple_choice_instances(df, split, label_cname)

    def get_instances(self, output_path: str) -> List[Instance]:
        self.data_dir = os.path.join(output_path, "data")

        # download dataset
        self.download_dataset()

        # download MITRE_ATT_CK_V10 information
        self.download_MITRE_info()

        # load dataset
        all_data_dir = os.path.join(self.data_dir, self.dataset_all_name)
        all_df = pd.read_csv(all_data_dir)

        # split all_df into train and test data frames
        train_df = all_df.sample(frac=CtiMitreScenario.train_ratio, random_state=0)
        test_df = all_df.drop(train_df.index).sample(frac=1, random_state=0)

        # load MITRE info json data
        label_name_json = os.path.join(
            self.data_dir, self.mitre_dir, self.enterprise_attack_dir, self.enterprise_attack_json
        )
        jdata = None
        with open(label_name_json) as f:
            jdata = json.load(f)

        # make mapping from label_tec to tec_category_name
        label_cname = self.make_label_category_name_dict(jdata)

        # create instances from each dataset
        instances_train = self.create_instances(train_df, TRAIN_SPLIT, label_cname)
        instances_test = self.create_instances(test_df, TEST_SPLIT, label_cname)

        # return all instances
        all_instances = []
        all_instances.extend(instances_train)
        all_instances.extend(instances_test)
        return all_instances
