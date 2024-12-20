import os
import sqlite3
import threading
from typing import Any, Optional, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class CzechBankQAAnnotator(Annotator):
    """The CzechBankQA autograder.

    MUST BE RUN WITH --num-threads 1 FOR SOME REASON"""

    name = "czech_bank_qa"

    DATABASE_SOURCE_URL = (
        "https://huggingface.co/datasets/yifanmai/czech_bank_qa/resolve/main/czech_bank.db?download=true"
    )

    def __init__(self, file_storage_path: str):
        super().__init__()

        cache_dir = os.path.join(file_storage_path, "data")
        ensure_directory_exists(cache_dir)
        file_name = "czech_bank.db"
        file_path = os.path.join(cache_dir, file_name)
        ensure_file_downloaded(source_url=CzechBankQAAnnotator.DATABASE_SOURCE_URL, target_path=file_path)
        database = sqlite3.connect(file_path)

        # csv_files_dir = "/home/yifanmai/oss/helm-scenarios/1999-czech-bank"
        # # table_name_to_file_name = {
        # #     "account": "account.csv",
        # #     "client": "client.csv",
        # #     "disposition": "disp.csv",
        # #     "permenant_order": "order.csv",
        # #     "transaction": "trans.csv",
        # #     "loan": "loan.csv",
        # #     "credit_card": "card.csv",
        # #     "demographic_data": "district.csv"
        # # }
        # for file_name in os.listdir(csv_files_dir):
        #     file_path = os.path.join(csv_files_dir, file_name)
        #     df = pd.read_csv(file_path)
        #     table_name = file_name.removesuffix(".csv")
        #     df.to_sql(table_name, database, if_exists="append", index=False)
        #     print("Commited to SQL")
        # # df.to_sql(table_name, conn, if_exists='append', index=False)

        self.database = database
        self.lock = threading.Lock()

    def get_result(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        result: Optional[str] = None
        error: Optional[str] = None
        try:
            cursor = self.database.cursor()
            cursor.execute("PRAGMA query_only = TRUE")
            cursor.execute(query)
            result = str(cursor.fetchall())
            cursor.close()
        except (sqlite3.DatabaseError, sqlite3.Warning) as e:
            error = str(e)
        return (result, error)

    def annotate(self, request_state: RequestState) -> Any:
        assert request_state.result
        assert len(request_state.result.completions) == 1

        assert len(request_state.instance.references) == 1
        gold_query = request_state.instance.references[0].output.text
        query = request_state.result.completions[0].text
        query = query.replace("```sql", "").replace("```", "")
        result, error = self.get_result(query)
        gold_result, gold_error = self.get_result(gold_query)

        return {"query": query, "result": result, "error": error, "gold_result": gold_result, "gold_error": gold_error}
