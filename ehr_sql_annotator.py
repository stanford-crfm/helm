from typing import Any, List, Optional
import os
import re
import sqlite3
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.common.hierarchical_logger import hlog
from helm.benchmark.runner import get_benchmark_output_path


class EhrSqlAnnotator(Annotator):
    """
    Executes both ground truth and generated SQL queries on the eicu.sqlite database.
    """

    name = "ehr_sql"

    def annotate(self, request_state: RequestState) -> Any:
        """Evaluate SQL execution accuracy by running queries against the eicu.sqlite database."""

        # Ensure the database file is correctly referenced
        database_path = os.path.join(get_benchmark_output_path(), "eicu.sqlite")

        # Validate ground truth query existence
        assert len(request_state.instance.references) == 1
        ground_truth_sql = request_state.instance.references[0].output.text.strip()
        ground_truth_result: List[str] = []

        # Execute the ground truth query
        try:
            with sqlite3.connect(database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(ground_truth_sql)
                ground_truth_result = cursor.fetchall()
        except (sqlite3.OperationalError, sqlite3.Warning) as e:
            hlog(f"WARNING: Ground truth SQL failed with error: {e}")

        # Ensure LLM-generated query exists
        assert request_state.result is not None
        assert len(request_state.result.completions) == 1
        predicted_text = request_state.result.completions[0].text.strip()

        # Extract SQL query from LLM output
        predicted_sql_match = re.search(r"<\s*sql\s*>(.*?)<\/?\s*sql\s*>", predicted_text, re.DOTALL | re.IGNORECASE)
        predicted_sql = predicted_sql_match.group(1).strip() if predicted_sql_match else predicted_text

        predicted_result: List[str] = []
        query_error: Optional[str] = None

        # Execute the predicted SQL query
        try:
            with sqlite3.connect(database_path) as conn:
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
