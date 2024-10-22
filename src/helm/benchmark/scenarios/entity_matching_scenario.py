import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from helm.common.hierarchical_logger import hlog
from helm.common.general import ensure_file_downloaded
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
from helm.benchmark.scenarios.entity_matching_scenario_fixed_random_state import set_fixed_random_state_for_dataset


class EntityMatchingScenario(Scenario):
    """
    Scenario for the entity matching task.

    This scenario supports all datasets from the benchmark entity matching datasets from
    https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md. We anticipate only
    running one from each category of Structured, Textual, and Dirty.

    Entity matching (EM) is a core preprocessing step in structured data ETL pipelines. The task
    is as follows. Given two structured relations A and B, determine which rows from each relation
    refer to the same underlying entity and which ones do not. Typically the task is separated
    into two steps. The first performs blocking (or candidate generation) where a small set of
    possible matches from B are generated for each row in A. The second does matching where
    for each row in A and candidate pair, generate a T/F label if the pair refers to the same entity.
    To make benchmarking performance easier, standard EM benchmarks come pre-blocked. Therefore,
    the goal is to simply determine which pairs are matches or not.

    A negative and positive example are below. Note that there are no newlines for a single row. We
    only add new lines between each row.

    Input

        Row A: title: adobe creative suite cs3 design premium upsell [ mac ] | manufacturer: adobe price: 1599.0
        Row B: title: 19600061dm adobe creative suite 3 production premium media tlp download mac world
        | manufacturer: nan price: 20.97

    Reference [CORRECT]: No

    Input

        Row A: title: adobe creative suite cs3 web premium upgrade [ mac ] | manufacturer: adobe price: 499.0
        Row B: title: adobe cs3 web premium upgrade | manufacturer: nan price: 517.99

    Reference [CORRECT]: Yes

    The above example highlights the model will need to reason over semantic dissimilarities (e.g.,
    premium upsell being in [A] but not [B] in the first example) as well as notions of price
    similarity (e.g., 499 is closer to 518 compared to 1599 versus 21.)
    """

    name = "entity_matching"
    description = "T/F generation task for determining if two structured rows refer to the same entity."
    tags = ["entity_matching", "structured_reasoning"]

    def __init__(self, dataset: str):
        super().__init__()
        self.em_datasets_paths = {
            "Beer": "Structured/Beer",
            "iTunes_Amazon": "Structured/iTunes-Amazon",
            "Fodors_Zagats": "Structured/Fodors-Zagats",
            "DBLP_ACM": "Structured/DBLP-ACM",
            "DBLP_GoogleScholar": "Structured/DBLP-GoogleScholar",
            "Amazon_Google": "Structured/Amazon-Google",
            "Walmart_Amazon": "Structured/Walmart-Amazon",
            "Abt_Buy": "Textual/Abt-Buy",
            "Company": "Textual/Company",
            "Dirty_iTunes_Amazon": "Dirty/iTunes-Amazon",
            "Dirty_DBLP_ACM": "Dirty/DBLP-ACM",
            "Dirty_DBLP_GoogleScholar": "Dirty/DBLP-GoogleScholar",
            "Dirty_Walmart_Amazon": "Dirty/Walmart-Amazon",
        }
        assert dataset in self.em_datasets_paths
        self.dataset = dataset

    def read_tables(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read in relations A and B."""
        tableA = pd.read_csv(data_path / "tableA.csv")
        tableB = pd.read_csv(data_path / "tableB.csv")
        return tableA, tableB

    def read_blocked_pairs(
        self, data_path: Path, split: str, tableA: pd.DataFrame, tableB: pd.DataFrame
    ) -> pd.DataFrame:
        """Read in pre-blocked pairs with T/F match labels."""
        labels = pd.read_csv(data_path / f"{split}.csv")

        mergedA = pd.merge(labels, tableA, right_on="id", left_on="ltable_id")
        merged = pd.merge(mergedA, tableB, right_on="id", left_on="rtable_id", suffixes=("_A", "_B"))
        # If train, downsample negative rows to make class balanced for prompt sampling
        if split == TRAIN_SPLIT:
            num_pos_classes: int = sum(merged["label"] == 1)
            num_neg_classes: int = sum(merged["label"] == 0)
            assert num_pos_classes < num_neg_classes
            sample_fn = lambda x: x.sample(num_pos_classes)
            merged = merged.groupby("label", group_keys=False).apply(sample_fn)
        return merged

    def serialize_row(self, row: pd.core.series.Series, column_map: Dict[str, str]) -> str:
        """Turn structured row into string."""
        res = []
        for c_og, c_map in column_map.items():
            res.append(f"{c_map}: {row[c_og]}".strip())
        return ". ".join(res)

    def get_instances(self, output_path: str) -> List[Instance]:
        set_fixed_random_state_for_dataset(output_path, self.dataset)
        data_path = Path(output_path) / "data" / self.dataset
        data_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_file_downloaded(
            source_url=f"http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/"
            f"{self.em_datasets_paths[self.dataset]}/"
            f"{self.dataset.lower()}_exp_data.zip",
            target_path=str(data_path),
            unpack=True,
            unpack_type="unzip",
        )
        # Read in tableA and tableB
        tableA, tableB = self.read_tables(data_path)
        # Columns for row serialization; do not add id column
        column_mapA = {f"{c}_A": c for c in tableA.columns if c != "id"}
        column_mapB = {f"{c}_B": c for c in tableB.columns if c != "id"}
        # Read in tables and matches
        labeled_examples = {
            "train": self.read_blocked_pairs(data_path, TRAIN_SPLIT, tableA, tableB),
            "valid": self.read_blocked_pairs(data_path, VALID_SPLIT, tableA, tableB),
            "test": self.read_blocked_pairs(data_path, TEST_SPLIT, tableA, tableB),
        }
        # Read all the instances
        instances: List[Instance] = []

        for split, example_df in labeled_examples.items():
            hlog(f"Processing {split} with {example_df.shape[0]} rows")
            for _, pair in example_df.iterrows():
                resA: str = self.serialize_row(pair, column_mapA)
                resB: str = self.serialize_row(pair, column_mapB)
                input: str = f"Product A is {resA}. Product B is {resB}. Are A and B the same?"
                label: str = "Yes" if pair["label"] else "No"

                instance = Instance(
                    input=Input(text=input),
                    references=[Reference(Output(text=label), tags=[CORRECT_TAG])],
                    split=split,
                )

                instances.append(instance)

        return instances
