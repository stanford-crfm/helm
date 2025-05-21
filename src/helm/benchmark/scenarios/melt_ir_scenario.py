from typing import List, Optional

from datasets import load_dataset, Dataset
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    make_rank_tag,
)


class MELTInformationRetrievalScenario(Scenario):
    name = "melt_information_retrieval"
    description = "Scenario for information retrieval tasks."
    tags = ["information_retrieval"]

    """ Dictionary mapping task track tuples to the number of queries. """
    NUM_TRAIN_QUERIES = 1000

    """ Upper and lower bounds on top-k.

    The top-k number represents the number of passages we will consider per
    query. Max top-k for the train and validation files are set to the number
    of passages included in the corresponding top-k files.
    """
    MIN_TOPK: int = 11
    MAX_TRAIN_TOPK: int = 20
    MAX_VALID_TOPK: int = 1000

    def __init__(
        self, dataset_name: str, revision: str, subset: Optional[str] = None, valid_topk: Optional[int] = None
    ):
        """The constructor for the MSMARCOScenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            subset: The subset of the dataset to use. Defaults to "".
            valid_topk: If set, specifies the number of top documents for which the
                validation instances will be created. Must be in the range
                [self.MIN_TOPK, self.MAX_VALID_TOPK].
        """
        super().__init__()

        # Input validation
        self.dataset_name = dataset_name
        self.revision = revision
        self.subset = subset
        self.valid_topk: Optional[int] = valid_topk
        if self.valid_topk is not None:
            assert valid_topk and self.MIN_TOPK <= valid_topk <= self.MAX_VALID_TOPK

    def get_train_instances(self) -> List[Instance]:
        """Get training instances.
        References for each instance are selected as follows:
            1. We select 1 correct reference, where the documents included
            corresponds to the best document for the given train query.
            2. We create 1 wrong reference, where the document included
            corresponds to a non-gold document for the given train query.
        """
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )
        instances = []
        for i, sample in enumerate(dataset["train"]):

            if i >= self.NUM_TRAIN_QUERIES:
                break

            references = [
                Reference(Output(text=sample["positive"]), tags=[CORRECT_TAG]),
                Reference(Output(text=sample["negative"]), tags=[]),
            ]

            instances.append(Instance(Input(text=sample["query"]), references=references, split=TRAIN_SPLIT))
        return instances

    def get_valid_instances(self) -> List[Instance]:
        """Get validation instances.
        By default, we create a reference for each Document ID for which there
        is a judgment with respect to the provided Query ID.

        If self.valid_topk is not None, we ensure that a reference is created
        for all the documents that appear in top self.valid_topk documents for
        the given validation query.
        """
        dataset = load_dataset(
            self.dataset_name,
            f"runs-{self.subset}",
            revision=self.revision,
            trust_remote_code=True,
        )
        instances = []
        for sample in dataset["bm25"]:
            references = []

            for k, passage_dict in enumerate(Dataset.from_dict(sample["passages"])):
                if self.valid_topk is None or k >= self.valid_topk:
                    break
                tags = []
                tags.append(f"docid={passage_dict['id']}")
                if k == 0:
                    tags.append(CORRECT_TAG)
                tags.append(make_rank_tag(rank=k + 1))  # Top-k rank
                references.append(Reference(Output(text=passage_dict["passage"]), tags=tags))

            instances.append(Instance(Input(text=sample["query"]), references=references, split=VALID_SPLIT))

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Get instances for this scenario.

        Refer to the documentation of the following methods for details on how
        the instances are created:
            * self.get_train_instances
            * self.get_valid_instances
        """

        hlog("Preparing training instances.")
        train_instances = self.get_train_instances()

        hlog("Preparing validation instances.")
        valid_instances = self.get_valid_instances()

        return train_instances + valid_instances


class MELTInformationRetrievalMMARCOScenario(MELTInformationRetrievalScenario):
    """
    Scenario for the MMARCO dataset.
    """

    name = "melt_information_retrieval_mmarco"
    description = "MMARCO dataset for information retrieval in Vietnamese."
    tags = ["information_retrieval"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="unicamp-dl/mmarco",
            revision="6d039c4638c0ba3e46a9cb7b498b145e7edc6230",
            subset="vietnamese",
            **kwargs,
        )


class MELTInformationRetrievalMRobustScenario(MELTInformationRetrievalScenario):
    """
    Scenario for the MRobust dataset.
    """

    name = "melt_information_retrieval_mrobust"
    description = "MRobust dataset for information retrieval in Vietnamese."
    tags = ["information_retrieval"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="unicamp-dl/mrobust",
            revision="fda452a7fbfd9550db2f78d9d98e6b3ec16734df",
            subset="vietnamese",
            **kwargs,
        )
