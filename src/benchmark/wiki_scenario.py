import csv
import os
from typing import List
from collections import defaultdict
import re
import pdb

from pyparsing import delimited_list
from transformers import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


class WIKIScenario(Scenario):
    """
    """

    name = "wiki"
    description = "Fact Completion in WikiData"
    tags = ["knowledge", "generation"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        # TODO: download the file from server

        # Read all the instances
        instances = []
        qid2name = defaultdict(list)
        with open(os.path.join(data_path, "names.tsv")) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                qid2name[row[0]].append(row[1])

        def answer_to_reference(answer):
            references = []
            if re.match("^[Q][0-9]+", answer) is not None:  # QID as answer, use qid2name to convert it to a string
                # assert answer in qid2name
                for name in qid2name[answer]:
                    references.append(Reference(output=" "+name+".", tags=[CORRECT_TAG]))
            else:
                references.append(Reference(output=answer, tags=[CORRECT_TAG]))
            return references

        def check_valid_references(references):
            return len(references) > 0

        sentence_ends = defaultdict(int)
        with open(os.path.join(data_path, "data.tsv")) as f:
            reader = csv.reader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                question, answer = row[0], row[1]
                if idx < 32227 or idx >= 32247:
                    continue
                references = answer_to_reference(answer)
                if not check_valid_references(references):
                    continue
                sentence_ends[question[-1]] += 1
                # if question[-1] != " ":
                #     question += " "
                if idx < 32232:
                    instance = Instance(input=question, references=references, tags=[TRAIN_TAG],)
                else:
                    instance = Instance(input=question, references=references, tags=[TEST_TAG],)
                instances.append(instance)
        return instances


if __name__ == "__main__":
    s = WIKIScenario()
    s.output_path = "benchmark_output/scenarios/wiki"
    i = s.get_instances()
    for ii in i:
        print('-'*10)
        print(ii)
    pdb.set_trace()
'''
1. 71555 queries in data.tsv
 70814 have answers with qid in names.tsv
2. Do we want to have space to be the end of a question? 
clean the data so that it ends with a space.
defaultdict(<class 'int'>, {'f': 7101, 'n': 7583, 's': 44727, 'e': 1787, 'y': 5011, 'r': 899, 'h': 2991, ' ': 622, 'm': 93})
'''