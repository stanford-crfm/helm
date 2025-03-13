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

        databases_root_path = os.path.join(get_benchmark_output_path(), "scenarios", "ehr_sql")
        database_path = os.path.join(databases_root_path, "eicu.sqlite")

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

        # If ground truth SQL execution didn't return results, attempt to use extra_data["value"]
        if not ground_truth_result and request_state.instance.extra_data is not None:
            if "value" in request_state.instance.extra_data:
                extra_values = list(request_state.instance.extra_data["value"].values())

                # Try inferring types from the database schema if possible
                with sqlite3.connect(database_path) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(ground_truth_sql)
                        fetched_result = cursor.fetchone()
                        if fetched_result:
                            # Convert extra_values to match SQLite's expected types
                            converted_values = [
                                type(fetched_result[i])(extra_values[i]) for i in range(len(extra_values))
                            ]
                            ground_truth_result = converted_values
                        else:
                            # If no rows were fetched, use `extra_values` as-is
                            ground_truth_result = extra_values
                    except sqlite3.OperationalError:
                        # If query fails (syntax error, etc.), just use `extra_values` as-is
                        ground_truth_result = extra_values

        assert request_state.result is not None
        assert len(request_state.result.completions) == 1
        predicted_text = request_state.result.completions[0].text.strip()

        predicted_sql_match = re.search(r"<\s*sql\s*>(.*?)<\/?\s*sql\s*>", predicted_text, re.DOTALL | re.IGNORECASE)
        predicted_sql = predicted_sql_match.group(1).strip() if predicted_sql_match else predicted_text.strip()

        predicted_result: List[str] = []
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
            except (sqlite3.OperationalError, sqlite3.Warning) as e:
                query_error = str(e)

        return {
            "predicted_result": predicted_result,
            "ground_truth_result": ground_truth_result,
            "query_error": query_error,
        }
