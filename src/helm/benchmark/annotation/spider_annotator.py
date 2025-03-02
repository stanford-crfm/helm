import os

from helm.benchmark.annotation.bird_sql_annotator import BirdSQLAnnotator
from helm.benchmark.runner import get_benchmark_output_path


class SpiderAnnotator(BirdSQLAnnotator):
    """The Spider evaluator that computes execution accuracy.

    Based on the Bird-SQL annotator."""

    name = "spider"

    def get_database_path(self, database_name: str) -> str:
        databases_root_path = os.path.join(
            get_benchmark_output_path(), "scenarios", "spider", "data", "spider_data", "test_database"
        )
        return os.path.join(databases_root_path, database_name, f"{database_name}.sqlite")
