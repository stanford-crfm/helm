from pathlib import Path

from typing import List, Optional, Any

import datasets
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)

_ALL_LANGUAGES = {
    "bulgarian": "bg",
    "czech": "cs",
    "dutch": "nl",
    "estonian": "et",
    "french": "fr",
    "greek": "el",
    "irish": "ga",
    "latvian": "lv",
    "maltese": "mt",
    "portuguese": "pt",
    "slovak": "sk",
    "spanish": "es",
    "croatian": "hr",
    "danish": "da",
    "english": "en",
    "finnish": "fi",
    "german": "de",
    "hungarian": "hu",
    "italian": "it",
    "lithuanian": "lt",
    "polish": "pl",
    "romanian": "ro",
    "slovenian": "sl",
    "swedish": "sv",
}

# IMPORTANT: So far, we only support English, since we would need to adapt the metrics for scoring the other languages.
_SUPPORTED_LANGUAGES = {
    "english": "en",
}


def _clean_and_truncate(text: str, max_num_words: Optional[int] = None) -> str:
    text = text.replace("\n", " ")
    whitespace_tokens = text.split()
    if max_num_words:
        whitespace_tokens = whitespace_tokens[:max_num_words]
    return " ".join(whitespace_tokens)


class LegalSummarizationScenario(Scenario):
    """
    Scenario for single document text summarization.
    Currently, supports the following datasets:
    1. BillSum (https://aclanthology.org/D19-5406/)
    2. MultiLexSum (https://arxiv.org/abs/2206.10883)
    3. EurLexSum (https://arxiv.org/abs/2210.13448)

    Task prompt structure

        Summarize the given document.
        Document: {tok_1 ... tok_n}
        Summary: {tok_1 ... tok_m}

    Example from MultiLexSum dataset (Short to Tiny)

        Document: {This case is about an apprenticeship test that had a disparate impact
                    on Black apprenticeship applicants. The Equal Employment Opportunity
                    Commission (EEOC) filed this lawsuit on December 27, 2004, in U.S.
                    District Court for the Southern District of Ohio. Filing on behalf
                    of thirteen Black individuals and a class of similarly situated Black
                    apprenticeship test takers, the EEOC alleged that the individuals’
                    employer, the Ford Motor Company, as well as their union, the United
                    Automobile, Aerospace, and Agricultural implement workers of America
                    (the “UAW”), and the Ford-UAW Joint Apprenticeship Committee, violated
                    Title VII of the Civil Rights Act, 42 U.S.C. § 1981, and Michigan state
                    anti-discrimination law. The EEOC sought injunctive relief and damages
                    for the Black apprenticeship applicants. The individuals also brought a
                    separate class action against Ford and the UAW, and the cases were
                    consolidated. In June 2005, both cases were resolved via a class
                    settlement agreement. Ford agreed to pay $8.55 million and to implement
                    a new selection process for its apprenticeship programs, and the court
                    ordered Ford to cover attorneys’ fees and expenses. This case is closed.}
        Summary: {2005 class action settlement resulted in Ford paying $8.55m to redesign
                    its selection process for apprenticeship programs to address the
                    previous process’s disparate impact on Black applicants.}
    """

    name = "summarization"
    description = "Scenario for legal summarization tasks"
    tags = ["summarization"]

    # Mapping from HELM splits to HF splits
    splits_mapping = {
        TRAIN_SPLIT: datasets.Split.TRAIN,
        VALID_SPLIT: datasets.Split.VALIDATION,  # BillSum does not have any validation split
        TEST_SPLIT: datasets.Split.TEST,
    }

    def __init__(
        self,
        dataset_name: str,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
    ):
        """
        Initializes summarization scenario.
        Args:
            dataset_name: String identifier for dataset. Currently
                          supported options ["BillSum", "MultiLexSum", "EurLexSum"].
            sampling_min_length: Int indicating minimum length (num whitespace-separated tokens) for training
                                 documents. Training examples smaller than
                                 sampling_min_length will be filtered out.
                                 Useful for preventing the adapter from sampling
                                 really small documents.
            sampling_max_length: Int indicating maximum length (num whitespace-separated tokens) for training
                                 documents. Training examples larger than
                                 sampling_max_length will be filtered out.
                                 Useful for preventing the adapter from
                                 sampling really large documents.
            doc_max_length: Int indicating the maximum length (num whitespace-separated tokens) to truncate
                            documents. Documents in all splits will be
                            truncated to doc_max_length tokens.
                            NOTE: Currently uses whitespace tokenization.
        """
        super().__init__()
        if dataset_name not in ["BillSum", "MultiLexSum", "EurLexSum"]:
            raise Exception(f"Unknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _download_dataset(self, hf_name, output_path):
        cache_dir = str(Path(output_path) / "data")
        dataset: Any
        if self.dataset_name == "EurLexSum":
            # IMPORTANT: change this once more languages should be supported
            dataset = load_dataset(hf_name, "english", cache_dir=cache_dir)
        elif self.dataset_name == "MultiLexSum":
            dataset = load_dataset(hf_name, "v20220616", cache_dir=cache_dir)
        else:
            dataset = load_dataset(hf_name, cache_dir=cache_dir)

        return dataset

    def _load_dataset(self, dataset_name: str, output_path):
        if dataset_name == "BillSum":
            hf_name = "billsum"
            dataset = self._download_dataset(hf_name, output_path)
            article_key = "text"
            summary_key = "summary"
        elif dataset_name == "MultiLexSum":
            hf_name = "allenai/multi_lexsum"
            dataset = self._download_dataset(hf_name, output_path)
            article_key = "summary/long"
            summary_key = "summary/short"  # we could do summary/tiny additionally here
        elif dataset_name == "EurLexSum":
            hf_name = "dennlinger/eur-lex-sum"
            dataset = self._download_dataset(hf_name, output_path)
            article_key = "reference"
            summary_key = "summary"
        else:
            raise ValueError("The specified dataset is not supported")

        return dataset, article_key, summary_key

    def get_instances(self, output_path) -> List[Instance]:
        dataset, article_key, summary_key = self._load_dataset(self.dataset_name, output_path)

        instances: List[Instance] = []
        for split, split_name in self.splits_mapping.items():
            if split_name not in dataset:
                continue  # BillSum does not have a validation split
            for example in dataset[split_name]:
                if not example[summary_key]:
                    continue  # skip empty summaries: In MultiLexSum some summaries are empty
                article: str = _clean_and_truncate(example[article_key], self.doc_max_length)
                summary: str = _clean_and_truncate(example[summary_key])

                if split_name == "train":
                    art_len = len(article.split())
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue

                instances.append(
                    Instance(
                        input=Input(text=article),
                        references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
