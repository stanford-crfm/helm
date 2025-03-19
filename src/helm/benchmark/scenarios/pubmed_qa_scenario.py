import json
import os
from typing import Dict, List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    ALL_SPLITS,
    CORRECT_TAG,
    Reference,
    PassageQuestionInput,
    Output,
)


class PubMedQAScenario(Scenario):
    """
    From "PubMedQA: A Dataset for Biomedical Research Question Answering" (Jin et al.),
    PubMedQA is a biomedical QA dataset collected from PubMed abstracts, where the answer to the questions are
    one of yes/no/maybe. We use the " PQA-L(abeled)" subset, which has 1,000 labeled question-answer pairs
    annotated by human experts.

    We generated the splits using the official script:
    https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py.
    The train and dev splits are from the "pqal_fold0" fold. A copy of the preprocessed dataset is stored at
    https://worksheets.codalab.org/bundles/0x531c9c54d8314d289da812af608b86fb.

    The following is an example from the dataset

    ```
    "QUESTION": "Is anorectal endosonography valuable in dyschesia?",
    "CONTEXTS": [
        "Dyschesia can be provoked by inappropriate defecation movements. The aim of this prospective study was to
        demonstrate dysfunction of the anal sphincter and/or the musculus (m.) puborectalis in patients with dyschesia
        using anorectal endosonography.",
        "Twenty consecutive patients with a medical history of dyschesia and a control group of 20 healthy subjects
        underwent linear anorectal endosonography (Toshiba models IUV 5060 and PVL-625 RT). In both groups, the
        dimensions of the anal sphincter and the m. puborectalis were measured at rest, and during voluntary squeezing
        and straining. Statistical analysis was performed within and between the two groups.",
        "The anal sphincter became paradoxically shorter and/or thicker during straining (versus the resting state) in
        85% of patients but in only 35% of control subjects. Changes in sphincter length were statistically
        significantly different (p<0.01, chi(2) test) in patients compared with control subjects. The m. puborectalis
        became paradoxically shorter and/or thicker during straining in 80% of patients but in only 30% of controls.
        Both the changes in length and thickness of the m. puborectalis were significantly different (p<0.01, chi(2)
        test) in patients versus control subjects."
    ],
    "LABELS": [
        "AIMS",
        "METHODS",
        "RESULTS"
    ],
    "MESHES": [
        "Adolescent",
        "Adult",
        "Aged",
        "Aged, 80 and over",
        "Anal Canal",
        "Case-Control Studies",
        "Chi-Square Distribution",
        "Constipation",
        "Defecation",
        "Endosonography",
        "Female",
        "Humans",
        "Male",
        "Middle Aged",
        "Pelvic Floor",
        "Rectum"
    ],
    "YEAR": "2002",
    "reasoning_required_pred": "yes",
    "reasoning_free_pred": "yes",
    "final_decision": "yes"
    ```

    Citation

    ```
    @inproceedings{jin2019pubmedqa,
      title={PubMedQA: A Dataset for Biomedical Research Question Answering},
      author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
      booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the
      9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
      pages={2567--2577},
      year={2019}
    }
    ```

    To reproduce the zero-shot performance of OpenAI's text-davinci-002 model on PubMedQA, we follow what was
    done in "Can large language models reason about medical questions?" (Liévin et al.) when constructing
    the `Instance`s.

    The following is the template of how they constructed the prompts

    ```
    Context: <Label>. <context>
    <Label>. <context>
    <Label>. <context>

    Question: <Question>

    A) yes
    B) no
    C) maybe
    ```

    among A through C, the answer is

    Citation

    ```
    @misc{https://doi.org/10.48550/arxiv.2207.08143,
      doi = {10.48550/ARXIV.2207.08143},
      url = {https://arxiv.org/abs/2207.08143},
      author = {Liévin, Valentin and Hother, Christoffer Egeberg and Winther, Ole},
      keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG),
      FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.1; I.2.7},
      title = {Can large language models reason about medical questions?},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }
    ```
    """

    name = "pubmed_qa"
    description = "A dataset that provides pubmed abstracts and asks associated questions yes/no/maybe questions."
    tags = ["question_answering", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["yes", "no", "maybe"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        url = (
            "https://raw.githubusercontent.com/pubmedqa/pubmedqa/"
            "1f00b98d5cc626844bf8c4ca513b6e62c40071ec/data/ori_pqal.json"
        )
        instances: List[Instance] = []
        for split in ALL_SPLITS:
            if split == "test":
                split_file_name: str = f"{split}_set.json"
                split_path: str = os.path.join(data_path, split_file_name)
                ensure_file_downloaded(
                    source_url=url,
                    target_path=split_path,
                    unpack=False,
                )

                with open(split_path, "r") as f:
                    split_examples: Dict = json.load(f)
                    for example in split_examples.values():
                        context_labels: List[str] = example["LABELS"]
                        contexts: List[str] = example["CONTEXTS"]
                        assert len(contexts) == len(context_labels)

                        # Format: <Label>. <context>
                        #         <Label>. <context>
                        # Example: Methods. Sixteen swine were used...
                        #          Results. Application of QC led to...
                        background: str = "\n".join(
                            [f"{label.title()}. {context}" for label, context in zip(context_labels, contexts)]
                        )

                        # Build `Reference`s. The possible answer choices are one of: "yes", "no" or "maybe"
                        correct_answer: str = example["final_decision"]
                        assert correct_answer in PubMedQAScenario.POSSIBLE_ANSWER_CHOICES
                        references: List[Reference] = [
                            Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])
                            for answer in PubMedQAScenario.POSSIBLE_ANSWER_CHOICES
                        ]

                        # Following Liévin et al., prepend the question with the provided context.
                        # Examples can be found here: https://vlievin.github.io/medical-reasoning/samples/pubmedqa.html.
                        question: str = example["QUESTION"]
                        prompt = PassageQuestionInput(
                            passage=background, question=question + "\n", passage_prefix="Context: ", separator="\n\n"
                        )
                        instance: Instance = Instance(input=prompt, references=references, split=split)
                        instances.append(instance)

        return instances
