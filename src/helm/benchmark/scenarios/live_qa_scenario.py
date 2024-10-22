import os
from typing import List
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference, Scenario


class LiveQAScenario(Scenario):
    """
    TREC-2017 LiveQA: Medical Question Answering Task

    The LiveQA'17 medical task focuses on consumer health question answering.
    Please refer to the original paper for more information about the constructed datasets and the LiveQA Track:
    https://trec.nist.gov/pubs/trec26/papers/Overview-QA.pdf

    Paper citation:

        @inproceedings{LiveMedQA2017,
          author    = {Asma {Ben Abacha} and Eugene Agichtein and Yuval Pinter and Dina Demner{-}Fushman},
          title     = {Overview of the Medical Question Answering Task at TREC 2017 LiveQA},
          booktitle = {TREC 2017},
          year      = {2017}
        }
    """

    SOURCE_REPO_URL = "https://raw.githubusercontent.com/abachaa/LiveQA_MedicalTask_TREC2017/master/TestDataset/"
    FILENAME = "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"

    name = "live_qa"
    description = "TREC-2017 LiveQA: Medical Question Answering Task"
    tags = ["knowledge", "generation", "question_answering", "biomedical"]

    def download_liveqa(self, path: str):
        """Download the XML file containing the questions & reference answers"""
        ensure_file_downloaded(
            source_url=os.path.join(self.SOURCE_REPO_URL, self.FILENAME),
            target_path=os.path.join(path, self.FILENAME),
            unpack=False,
        )

    @staticmethod
    def remove_whitespace(s: str) -> str:
        """Just remove all whitespace from a string"""
        return " ".join(s.strip().split())

    @staticmethod
    def _extract_question_id(element: Element):
        return element.attrib["qid"]

    @classmethod
    def _extract_question(cls, element: Element) -> str:
        """Given an XML Element representing a question, extract just the question as text"""
        return cls.remove_whitespace(element.find("NLM-Summary").text)  # type: ignore

    @classmethod
    def _extract_answers(cls, element: Element) -> List[str]:
        """Given an XML Element representing a question, extract the reference answers"""
        answers = []
        for answer in element.iter("ANSWER"):
            answers.append(cls.remove_whitespace(answer.text))  # type: ignore

        return answers

    def process_xml(self, base_path: str) -> List[Instance]:
        """Parse the XMLs into question-answer(s) pairs"""
        xml_path = os.path.join(base_path, self.FILENAME)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        instances = []
        for question_root in root:
            # get the actual question and question ID
            id = self._extract_question_id(question_root)
            question = Input(self._extract_question(question_root))

            # parse out the reference answers
            answers = self._extract_answers(question_root)
            references = [Reference(Output(answer), tags=[CORRECT_TAG]) for answer in answers]

            # stitch it all together
            instances.append(Instance(question, references, split=TEST_SPLIT, id=id))

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """entrypoint to creating this scenario's instances"""
        # get the dataset
        self.download_liveqa(output_path)

        # get the instances by parsing the XML
        instances = self.process_xml(output_path)
        return instances
