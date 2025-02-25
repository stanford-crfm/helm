from typing import Any, List, Optional
import os
import re
import sqlite3

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.benchmark.runner import get_benchmark_output_path
from helm.common.hierarchical_logger import hlog


class BirdSQLAnnotator(Annotator):
    """The Bird-SQL evaluator that computes execution accuracy."""

    name = "bird_sql"

    def get_database_path(self, database_name: str) -> str:
        databases_root_path = os.path.join(
            get_benchmark_output_path(), "scenarios", "bird_sql", "dev", "unzipped_dev_databases", "dev_databases"
        )
        return os.path.join(databases_root_path, database_name, f"{database_name}.sqlite")

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.instance.extra_data is not None
        database_name = request_state.instance.extra_data["db_id"]
        print(self.get_database_path(database_name))
        conn = sqlite3.connect(self.get_database_path(database_name))

        assert len(request_state.instance.references) == 1
        ground_truth_sql = request_state.instance.references[0].output.text
        ground_truth_result: List[str] = []
        try:
            cursor = conn.cursor()
            cursor.execute(ground_truth_sql)
            ground_truth_result = cursor.fetchall()
        except (sqlite3.OperationalError, sqlite3.Warning) as e:
            hlog(f"WARNING: Ground truth SQL failed with error: {e}")

        assert request_state.result is not None
        assert len(request_state.result.completions) == 1
        predicted_text = request_state.result.completions[0].text
        predicted_sql_match = re.search(r"<\s*sql\s*>(.*?)<\/?\s*sql\s*>", predicted_text, re.DOTALL | re.IGNORECASE)
        predicted_sql = predicted_sql_match.group(1) if predicted_sql_match else ""
        predicted_result: List[str] = []
        query_error: Optional[str] = None
        # TODO: Run SQL queries with a timeout
        try:
            cursor = conn.cursor()
            cursor.execute(predicted_sql)
            predicted_result = cursor.fetchall()
        except (sqlite3.OperationalError, sqlite3.Warning) as e:
            query_error = str(e)

        return {
            "predicted_result": predicted_result,
            "ground_truth_result": ground_truth_result,
            "query_error": query_error,
        }
