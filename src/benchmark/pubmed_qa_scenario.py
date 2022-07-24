import json
import os
from typing import Dict, List

from common.general import ensure_directory_exists, ensure_file_downloaded
from .scenario import Scenario, Instance, ALL_SPLITS, CORRECT_TAG, Reference


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

    @inproceedings{jin2019pubmedqa,
      title={PubMedQA: A Dataset for Biomedical Research Question Answering},
      author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
      booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the
      9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
      pages={2567--2577},
      year={2019}
    }

    To reproduce the zero-shot performance of OpenAI's text-davinci-002 model on PubMedQA, we follow what was
    done in "Can large language models reason about medical questions?" (Liévin et al.) when constructing
    the `Instance`s.

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
    """

    name = "pubmed_qa"
    description = "A biomedical question answering (QA) dataset collected from PubMed abstracts."
    tags = ["question_answering", "biomedical"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        for split in ALL_SPLITS:
            split_file_name: str = f"{split}_set.json"
            split_path: str = os.path.join(data_path, split_file_name)
            ensure_file_downloaded(
                source_url="https://worksheets.codalab.org/rest/bundles/0x531c9c54d8314d289da812af608b86fb/"
                f"contents/blob/{split_file_name}",
                target_path=split_path,
                unpack=False,
            )

            with open(split_path, "r") as f:
                split_examples: Dict = json.load(f)
                for example in split_examples.values():
                    contexts: List[str] = example["CONTEXTS"]
                    question: str = example["QUESTION"]
                    answer: str = example["final_decision"]  # One of "yes", "no" or "maybe"

                    # Following Liévin et al., prepend the question with the provided context
                    # TODO: find an example of their prompt for PubMedQA and compare to our prompts.
                    #       The link to their code and example prompts (https://vlievin.github.io/medical-reasoning)
                    #       is currently broken.
                    instance: Instance = Instance(
                        input=" ".join(contexts + [question]),
                        references=[Reference(output=answer, tags=[CORRECT_TAG])],
                        split=split,
                    )
                    print(instance.input)
                    instances.append(instance)

        return instances
