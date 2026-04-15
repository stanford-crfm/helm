from typing import Any, List, Optional, Tuple
import os
import re
import sqlite3
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.common.hierarchical_logger import hwarn
from helm.benchmark.runner import get_benchmark_output_path


class EhrSqlAnnotator(Annotator):
    """
    Executes both ground truth and generated SQL queries on the eicu.sqlite database.
    """

    name = "ehr_sql"

    def annotate(self, request_state: RequestState) -> Any:
        """Evaluate SQL execution accuracy by running queries against the eicu.sqlite database."""

        databases_root_path = os.path.join(get_benchmark_output_path(), "scenarios", "ehr_sql")
        database_path = os.path.join(databases_root_path, "eicu.sqlite")

        assert len(request_state.instance.references) == 1
        ground_truth_sql = request_state.instance.references[0].output.text.strip()
        ground_truth_result: List[Tuple[Any, ...]] = []

        # Execute the ground truth query
        try:
            with sqlite3.connect(database_path) as conn:
                cursor = conn.cursor()

                # check the current_time in the ground_truth_sql
                if "current_time" in ground_truth_sql.lower():
                    # Replace current_time with the fixed timestamp
                    ground_truth_sql = re.sub(
                        r"current_time",
                        "'2105-12-31 23:59:00'",
                        ground_truth_sql,
                        flags=re.IGNORECASE,
                    )
                cursor.execute(ground_truth_sql)
                ground_truth_result = cursor.fetchall()
        except (sqlite3.OperationalError, sqlite3.Warning) as e:
            hwarn(f"Ground truth SQL failed with error: {e}")

        assert request_state.result is not None
        assert len(request_state.result.completions) == 1
        predicted_text = request_state.result.completions[0].text.strip()

        predicted_sql_match = re.search(r"<\s*sql\s*>(.*?)<\/?\s*sql\s*>", predicted_text, re.DOTALL | re.IGNORECASE)
        predicted_sql = predicted_sql_match.group(1).strip() if predicted_sql_match else predicted_text.strip()

        predicted_result: List[Tuple[Any, ...]] = []
        query_error: Optional[str] = None
        predicted_sql = predicted_sql.replace("`", "").strip()
        predicted_sql = re.sub(r"^sql\n", "", predicted_sql, flags=re.MULTILINE)
        if not predicted_sql:
            query_error = "No query generated"
        else:
            try:
                with sqlite3.connect(database_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(predicted_sql)
                    predicted_result = cursor.fetchall()
            except Exception as e:
                query_error = str(e)

        return {
            "predicted_result": predicted_result,
            "ground_truth_result": ground_truth_result,
            "query_error": query_error,
        }
