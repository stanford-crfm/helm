from io import StringIO
import pandas as pd
from pathlib import Path
from typing import List, Tuple

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


class EntityDataImputationScenario(Scenario):
    """
    Scenario for the entity data imputation task.

    This scenario supports the Restaurant and Buy datasets from
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9458712. We process the datasets
    as explained in the paper: remove all rows with NaN values and manually select 10% of the rows
    to serve as a test set. A categorical column from this set will be imputed. 20% of the remaining data
    is validation data.

    Entity data imputation is a core preprocessing step in structured data ETL pipelines. The task
    is as follows. Given a structured relation A with possible incomplete/NaN call values,
    determine what value should be used to fill in the cell. This task has traditionally relied
    on relational dependencies (e.g., if city = 'San Francisco' then state = 'CA') to infer missing
    values or some ML model trained to learn dependencies between cell values.

    An example is

        Input: title: adobe creative suite cs3 design premium upsell [ mac ] | price: 1599.0 | manufacturer:

    Reference [CORRECT]: adobe

    The above example highlights the model will need to reason over input titles and possibly external
    knowledge (e.g. that creative suite is from adobe) to generate the answer.
    """

    name = "entity_data_imputation"
    description = "Missing cell value generation task from a structured row."
    tags = ["entity_data_imputation", "structured_reasoning"]

    def __init__(self, dataset: str, seed: int = 1234):
        super().__init__()
        self.datasets_paths = {
            "Buy": (
                "https://storage.googleapis.com/crfm-helm-public/source_datasets/scenarios/"
                "entity_data_imputation/Abt-Buy.zip"
            ),
            "Restaurant": (
                "https://storage.googleapis.com/crfm-helm-public/source_datasets/scenarios/"
                "entity_data_imputation/restaurant.tar.gz"
            ),
        }
        # Columns to impute
        self.datasets_impute_col = {
            "Buy": "manufacturer",
            "Restaurant": "city",
        }
        assert dataset in self.datasets_paths
        self.dataset = dataset
        self.seed = seed

    def read_tables(self, data_path: Path) -> pd.DataFrame:
        """Read in structured data from given dataset."""
        if self.dataset == "Buy":
            table = pd.read_csv(data_path / "Buy.csv")
            # Remove unique ID column
            del table["id"]
            # To get the shape values as in the paper, we drop nan descriptions
            table = table[~table["description"].isna()]
        elif self.dataset == "Restaurant":
            columns = []
            data = []
            # Manual loading of text arff data (scipy doesn't support yet)
            in_data = False
            for line in open(data_path / "fz.arff"):
                if len(line.strip()) <= 0:
                    continue
                if in_data:
                    data.append(line.strip())
                if line.startswith("@"):
                    if line.startswith("@attribute"):
                        _, name, _ = line.split(maxsplit=2)
                        columns.append(name)
                    if line.startswith("@data"):
                        in_data = True
            table = pd.read_csv(StringIO("\n".join(data)), header=None, sep=",", names=columns)
            # Remove unique id
            del table["class"]
            # Remove explicit quote marks that carried over from the arff file
            for c in table.columns:
                table[c] = table[c].str.replace('"', "").str.strip()
            table = table.dropna()
        else:
            raise ValueError(f"{self.dataset} not recognized.")
        # Remove all rows with NaN values - we will manually add them back later
        return table

    def get_splits(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate train, valid, test splits.

        Following https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9458712,
        we select 10% of rows to be our test set. One column of this data will be
        imputed. Of the remaining 90% of the data, 20% is validation and 80% is training.
        """
        # Split into datasets
        test = table.sample(frac=0.1, random_state=self.seed)
        remaining = table.drop(test.index)
        valid = remaining.sample(frac=0.2, random_state=self.seed)
        train = remaining.drop(valid.index)
        return train, valid, test

    def serialize_row(self, row: pd.core.series.Series, columns: List[str]) -> str:
        """Turn structured row into string."""
        res = []
        for c in columns:
            res.append(f"{c}: {row[c]}".strip())
        return ". ".join(res)

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = Path(output_path) / "data" / self.dataset
        data_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_file_downloaded(
            source_url=self.datasets_paths[self.dataset],
            target_path=str(data_path),
            unpack=True,
        )
        # Read in structured table
        table = self.read_tables(data_path)
        # Get train, valid, test splits
        train, valid, test = self.get_splits(table)
        col_to_impute = self.datasets_impute_col[self.dataset]
        columns_for_serialize = [c for c in train.columns if c != col_to_impute]
        all_splits = {
            TRAIN_SPLIT: train,
            VALID_SPLIT: valid,
            TEST_SPLIT: test,
        }
        # Read all the instances
        instances: List[Instance] = []

        for split, example_df in all_splits.items():
            hlog(f"Processing {split} with {example_df.shape[0]} rows")
            for _, row in example_df.iterrows():
                res: str = self.serialize_row(row, columns_for_serialize)
                input: str = f"{res}. {col_to_impute}?"
                label = str(row[col_to_impute])
                instance = Instance(
                    Input(text=input), references=[Reference(Output(text=label), tags=[CORRECT_TAG])], split=split
                )
                instances.append(instance)

        return instances
