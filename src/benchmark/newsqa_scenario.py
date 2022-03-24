import os
import json
import random
from typing import List, Tuple
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, CORRECT_TAG


class NewsQAScenario(Scenario):
    """
    The NewsQA dataset is from the paper:
        https://arxiv.org/abs/1611.09830
    Original repository can be found at:
        https://github.com/Maluuba/newsqa
    Note: The training dataset cannot be directly shared due to copyright issues, and needs to be downloaded by
    following the instructions in the repo above. These instructions are duplicated here for
    convenience.
    1. Clone the repo (https://github.com/Maluuba/newsqa)
    2. Download the data from (https://msropendata.com/datasets/939b1042-6402-4697-9c15-7a28de7e1321).
    You need to create a login account to download this data.
    3. Download the CNN stories tar file from "https://cs.nyu.edu/~kcho/DMQA/"
    4. Create the conda environment using the command (conda create --name newsqa python=2.7 "pandas>=0.19.2")
    5. Install the requirements (conda activate newsqa && pip install --requirement requirements.txt)
    This should result in the creation of the file (combined-newsqa-data-v1.json) in the repo
    which is used in this scenario.
    NewsQA is a QA dataset containing 12,744 stories,
    and over 119,633 question-answer pairs. There are 92549 training qa pairs,
    5166 qas in the dev set, and 5126 in the test set.
    Particularly, given the a news article from CNN,
    the goal is answer questions with answers consisting of spans of text from the corresponding articles.
    All of the questions and answers are written by crowd sourced human annotators.
    For more details, see https://arxiv.org/abs/1611.09830.
    More concretely, we prompt models using the following format:
        Article: <news article>
        Question: <question>
        Answer:
        Target completion:
            <answer>
    Note: Some of the questions do not have an answer in the context so the
    model needs to answer "No Answer". While this behavior might be tricky to
    learn in the few-shot setting, we still include these examples in the
    scenario.
    Using an example from the training dataset, we have:
    NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted a wealthy businessman
    facing the death sentence for the killing of a teen in a case dubbed 'the house of horrors.'
    Moninder Singh Pandher was sentenced to death by a lower court in February...
    Question: Who was sentenced to death in February?
    Answer:
    References:
    ['Moninder Singh Pandher']
    """

    name = "newsqa"
    description = "Question answering using news articles."
    tags = ["question_answering"]

    def __init__(self):
        pass

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an sample from the dataset, create the prompt and the list of
        correct references.
        """
        prompt: str = ""
        prompt += f"Article: {sample['text']}\n\n"
        all_questions = sample["questions"]
        question = random.sample(all_questions, 1)[0]
        prompt += f"Question: {question['q']}\n"
        prompt += "Answer:"
        # generate set of valid answers
        answers = []

        # add the answer with consensus
        # two checks below since the key "noAnswer" is not always present in the dictionary question["consensus"],
        # and when it is present it is not always True
        if ("noAnswer" in question["consensus"].keys()) and (question["consensus"]["noAnswer"] is True):
            answers.append("No Answer")
        else:
            start_point = question["consensus"]["s"]
            end_point = question["consensus"]["e"]
            answer_text = sample["text"][start_point:end_point]
            answers.append(answer_text)
        # adding other crowdworker answers
        for answer in question["answers"]:
            if "noAnswer" in answer["sourcerAnswers"][0].keys():
                answer_text = "No Answer"
                # add to valid set of answers if it is already not present in the list
                if answer_text not in answers:
                    answers.append(answer_text)
            else:
                start_point = answer["sourcerAnswers"][0]["s"]
                end_point = answer["sourcerAnswers"][0]["e"]
                answer_text = sample["text"][start_point:end_point]
                if answer_text not in answers:
                    answers.append(answer_text)
        return prompt, answers

    def cleaned_samples(self, samples: List[dict]) -> List[dict]:
        """
        Given the full dataset this function only retains news article and QAs where there are
        at least one question that is valid. The question is valid if all crowdworkers believe that
        the question is valid and that the answer is present in text.
        """
        clean_samples: List = []
        for sample in samples:
            # set of valid questions in the sample
            valid_questions = []
            for question in sample["questions"]:
                add_question = True
                if ("isQuestionBad" in question.keys()) and (question["isQuestionBad"] != 0.0):
                    add_question = False
                if ("isAnswerAbsent" in question.keys()) and (question["isAnswerAbsent"] != 0.0):
                    add_question = False
                if ("badQuestion" in question["consensus"].keys()) and (question["consensus"]["badQuestion"] is True):
                    add_question = False
                if add_question is True:
                    valid_questions.append(question)
            clean = len(valid_questions) >= 1
            sample["questions"] = valid_questions
            if clean is True:
                clean_samples.append(sample)
        return clean_samples

    def get_file_instances(self, target_file: str, splits: dict) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            target_file (str): Data file.
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        file_instances: List[Instance] = []
        all_samples: List = []
        clean_samples: List = []
        with open(target_file, encoding="utf-8") as f:
            all_samples = json.load(f)["data"]
        clean_samples = self.cleaned_samples(all_samples)
        for sample in clean_samples:
            prompt, answers = self.create_prompt(sample)
            split = "train" if sample["type"] == "train" else "valid"
            instance = Instance(
                input=prompt,
                references=[Reference(output=ans, tags=[CORRECT_TAG]) for ans in answers],
                split=splits[split],
            )
            file_instances.append(instance)
        return file_instances

    def get_instances(self) -> List[Instance]:
        # TODO how to deal with NewsQA file. The dataset cannot be directly downloaded
        # currently hardcoded a path to a directory with the file
        data_path: str = "/nlp/scr/niladri/datasets/newsqa/"
        file_path = os.path.join(data_path, "combined-newsqa-data-v1.json")
        assert os.path.exists(file_path)
        splits = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT}
        random.seed(0)  # randomness needed to pick question at random
        instances: List[Instance] = self.get_file_instances(target_file=file_path, splits=splits)
        return instances
