import csv
import os
import re
from typing import Dict, List
import pyreadstat
from nltk import pos_tag
import numpy as np
import pandas as pd

from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


MCQ_TERMS = {"each of the following": False, 'that...': False}
REPLACE_DICT = {"Summary variable for NATPROBS biggest problem in the country today":
        "The biggest problem in the country today is...",
        " – even if neither is exactly right": "",
        " -- even if neither is exactly right": "",
        "’": "'"}

class PewScenario(Scenario):
    name = "pew_surveys"
    description = "Pew surveys"
    tags = ["multiple_choice"]

    def __init__(self, survey: str):
        self.survey: str = survey
        self.output_path = '../benchmarking/benchmark_output/scenarios/pew/'
        
    def first_lower(self, input_string):
        string_split = input_string.split(' ')
        if pos_tag(string_split[:1])[0][1] != 'NNP':
            string_split[0] = string_split[0].lower()
        return ' '.join(string_split)

    def split_string(self, input_string):

        split = input_string.split()
        sentences = [None]

        for si, s in enumerate(split):
            if sentences[-1] is None:
                sentences[-1] = s
            else:
                sentences[-1] += f" {s}"

            if s[-1] in ['.', '?', '!'] and si != len(split) - 1: 
                sentences.append(None)

        return sentences

    def make_ascii(self, text):
        text_output = ''.join([i if ord(i) < 128 else ' ' for i in text])
        if text != text_output:
            print(f"Ascii alert: Modifying text in dataset from {text} -> {text_output}")
        return text_output

    def replace_term(self, string, term_new):

        fixed = False
        string = string
        for term_old, mode in MCQ_TERMS.items():

            for term in [term_old.lower(), term_old.capitalize()]:
                if fixed: break
                if term in string:
                    fixed = True
                    if mode:
                        string = string.replace(term, term + ' ' + term_new)
                    else:
                        string = string.replace(term, term_new)
                    if f"{term_new} are" in string and pos_tag([term_new])[0][0] != 'NNS':
                            string = string.replace(f"{term_new} are", f"{term_new} is")
                    elif f"Are {term_new}" in string and pos_tag([term_new])[0][0] != 'NNS':
                        string = string.replace(f"Are {term_new} are", f"Is {term_new}")
        if not fixed: 
            return ""
        return string


    def post_process_question(self, question_key, question):

        for r in REPLACE_DICT:
            question = question.replace(r, REPLACE_DICT[r])


        question = question.replace('…', '...')

        qsplit = self.split_string(question)

        if '_W' in qsplit[0]:
            qsplit = qsplit[1:]
            question = ' '.join(qsplit).strip()

        if any([t in question for t in MCQ_TERMS]):
            suffix = self.first_lower(qsplit[-1])


            if 'that...' in question: suffix = f"that {suffix}..."

            qsplit = qsplit[:-1]
            question = self.replace_term(' '.join(qsplit).strip(), suffix)

        if len(qsplit) > 1 and 'Summary variable' in qsplit[-1]:
            qsplit = qsplit[:-1]
            question = ' '.join(qsplit).strip()


        return question

    def identity_valid_questions(self, key, value):
        if type(value) == str and re.search('Q\d.', value) is not None:
            return True
        if key in value and not any([s in key for s in ['LANG_', 'FORM_']]):
            return True
        return False

    def identify_valid_metadata(self, key, value):
        if 'F_' in key:
            return True
        return False


    def read_raw_data(self):
        questions, metadata, others = ({'key': [], 'value': [], 'options': []} for i in range(3))

        for k, v in self.info.column_names_to_labels.items():

            ops = list(self.info.variable_value_labels[k].values()) if k in self.info.variable_value_labels else []

            if self.identity_valid_questions(k, v):
                questions['key'].append(k)
                questions['value'].append(v)
                questions['options'].append(ops)
            elif self.identify_valid_metadata(k, v):
                metadata['key'].append(k)
                metadata['value'].append(v)
                metadata['options'].append(ops)
            else:
                others['key'].append(k)
                others['value'].append(v)
                others['options'].append(ops)

        return pd.DataFrame(questions), pd.DataFrame(metadata), pd.DataFrame(others)
    
    def preprocess_data(self, sav_path):
        self.responses, self.info = pyreadstat.read_sav(sav_path, 
                                                        apply_value_formats=True, 
                                                        formats_as_category=True, 
                                                        formats_as_ordered_category=False)
        
        raw_questions, self.metadata, self.others = self.read_raw_data()
        
        self.questions, self.question_to_options, self.multiple_ans_valid = [], {}, {}
        self.question_to_key = {}
    
        
        for idx, (k, q, o) in enumerate(zip(raw_questions['key'], raw_questions['value'], raw_questions['options'])):
            qpp = self.make_ascii(self.post_process_question(k, q))
            
            options = []
            for opt in o:
                for r in REPLACE_DICT:
                    opt = opt.replace(r, REPLACE_DICT[r])
                options.append(self.make_ascii(opt))
            
            if len(qpp) == 0 or len(o) <= 1: continue
            self.questions.append(qpp)
            self.question_to_options[qpp] = options
            self.multiple_ans_valid[qpp] = False
            self.question_to_key[qpp] = k
            assert all([r in o or np.isnan(r) for r in set(self.responses[k].values)])

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(self.output_path, "data")
        #ensure_file_downloaded(
        #    source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        #    target_path=data_path,
        #    unpack=True,
        #    unpack_type="untar",
        #)

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "auxiliary_train": TRAIN_SPLIT,
            "dev": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }
           
        sav_path: str = os.path.join(data_path, f"{self.survey}.sav")
            
        if not os.path.exists(sav_path):
            hlog(f"{sav_path} doesn't exist, skipping")

        hlog(f"Reading {sav_path}")
        self.preprocess_data(sav_path)
                
        for split in splits:
            
            for question in self.questions:
                answers = self.question_to_options[question]
                
                answers_dict = dict(zip(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], answers))
                correct_answer = answers[0] #Ignore, there is no correct answer

                def answer_to_reference(answer):
                    return Reference(output=answer, tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=question, references=list(map(answer_to_reference, answers)), split=splits[split],
                )
                instances.append(instance)

        return instances