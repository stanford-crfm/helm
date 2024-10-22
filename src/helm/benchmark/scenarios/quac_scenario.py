import json
import os
import random
from typing import List, Tuple

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class QuACScenario(Scenario):
    """
    The QuAC dataset is from the paper:
    https://arxiv.org/abs/1808.07036

    The original webpage is:
    http://quac.ai/

    QuAC is a QA dataset based on student-teacher dialogue. The student is shown
    the title and first paragraph of a Wikipedia page and tries to learn
    information about a section of the page.
    The training set contains 83,568 questions (11,567 dialogues), while the
    validation set contains 7,354 questions (1,000 dialogues).

    In this Scenario, we show the model all the relevant information (title,
    background, section title, section text) as well as a prefix of the dialogue
    and ask for the answer. Each dialogue contains between 4 and 12 questions so
    we randomly pick a stopping point to query the model (ensuring that at least
    two question-answer pairs are provided. Answers are at most 30 words long.

    For the validation set, there are 4 additional answers collected
    independently from other annotators (total 5 answers). Following the
    original paper, we treat all these answers as equally correct and compute
    the maximum F1 score of the model with respect to any of these answers.

    Concretely, we prompt models using the following format:

        Title: <title>
        Background: <first wiki paragraph>
        Section: <section title>
        Context: <section text>

        Question: <question_1>
        Answer: <answer_1>

        Question: <question_2>
        Answer: <answer_2>

        ...

        Question: <question_k>
        Answer:

        Target completion:
            <answer>

    Note: Some of the questions do not have an answer in the context so the
    model needs to answer "CANNOTANSWER". While this behavior might be tricky to
    learn in the few-shot setting, we still include these examples in the
    scenario.

    Example

    ```
    Title: Augusto Pinochet

    Background: Augusto Jose Ramon Pinochet Ugarte (; Spanish: [au'gusto
    pino'(t)Se, -'(t)Set]; 25 November 1915 - 10 December 2006) was a Chilean
    general, <...>

    Section: Accusations of fascism
    Context: Pinochet and his government have been characterised as fascist. For
    example, journalist and author Samuel Chavkin, in his book Storm Over Chile:
    The Junta Under Siege, <...>

    Question: What were the accusations?
    Answer: Griffin included Pinochet in a group of pseudo-populist despots
    distinct from fascism and including the likes of Saddam Hussein, Suharto,
    and Ferdinand Marcos.

    Question: What he accused of being a fascist?
    Answer: Pinochet attempted to build true fascism, the regime would likely
    have been toppled or at least been forced to alter its relationship to the
    United States.

    Question: Was there conflict because of his views?
    Answer: Anna Cento Bull also excluded Pinochet from fascism, although she
    has argued that his regime belongs to a strand of Cold War anti-communism

    Question: Is there something else interesting to know?
    Answer:
    ```

    Reference

    ```
    ["It is notable that in all the declarations of Pinochet's men, nobody has
    mentioned the creators of the new Chilean society and state,"]
    ```

    """

    name = "quac"
    description = "Question answering from dialog prompts."
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        prompt = ""
        prompt += f"Title: {sample['title']}\n\n"
        prompt += f"Background: {sample['background']}\n\n"

        prompt += f"Section: {sample['section_title']}\n"
        dialogue = sample["paragraphs"][0]

        context = dialogue["context"]
        assert context[-13:] == " CANNOTANSWER"
        context = context[:-13]
        prompt += f"Passage: {context}\n\n"

        qas = dialogue["qas"]
        num_qas = len(qas)
        k = random.randint(3, num_qas) - 1  # allow at least two QAs in dialogue

        for i in range(k - 1):
            prompt += f"Question: {qas[i]['question']}\n"
            prompt += f"Answer: {qas[i]['orig_answer']['text']}\n\n"

        prompt += f"Question: {qas[k]['question']}"

        answers = [ans["text"] for ans in qas[k]["answers"]]
        # De-duplicate the list with dict.fromkeys, which preserves the list order
        answers = list(dict.fromkeys(answers))

        if answers == ["CANNOTANSWER"]:
            answers.extend(["Not enough information", "Cannot answer", "Do not know"])

        return prompt, answers

    def get_split_instances(self, split_file: str, split: str) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            split_file (str): File for the corresponding data split
            split (str): Which split the instances are coming from.

        Returns:
            List[Instance]: Instances for the specified split
        """
        split_instances: List[Instance] = []

        with open(split_file, encoding="utf-8") as f:
            all_samples = json.load(f)["data"]

        for sample in all_samples:
            prompt, answers = self.create_prompt(sample)

            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=split,
            )
            split_instances.append(instance)

        return split_instances

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        base_url: str = "https://s3.amazonaws.com/my89public/quac"

        instances: List[Instance] = []
        splits = {"train": TRAIN_SPLIT, "val": VALID_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        for split, split_tag in splits.items():
            source_url: str = f"{base_url}/{split}_v0.2.json"
            split_path: str = os.path.join(data_path, f"{split}.json")
            ensure_file_downloaded(source_url=source_url, target_path=split_path)

            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances
