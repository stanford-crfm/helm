import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


class CzechBankQAScenario(Scenario):
    INSTRUCTIONS = """Given a SQLite database schema and the following instructions, generate a SQLite query that corresponds to the instructions. Answer with only the query.

Database schema:
CREATE TABLE "account" (
  "account_id" integer NOT NULL DEFAULT '0'
,  "district_id" integer NOT NULL DEFAULT '0'
,  "frequency" varchar(18) NOT NULL
,  "date" date NOT NULL
,  PRIMARY KEY ("account_id")
,  CONSTRAINT "account_ibfk_1" FOREIGN KEY ("district_id") REFERENCES "district" ("district_id")
);
CREATE TABLE "card" (
  "card_id" integer NOT NULL DEFAULT '0'
,  "disp_id" integer NOT NULL
,  "type" varchar(7) NOT NULL
,  "issued" date NOT NULL
,  PRIMARY KEY ("card_id")
,  CONSTRAINT "card_ibfk_1" FOREIGN KEY ("disp_id") REFERENCES "disp" ("disp_id")
);
CREATE TABLE "client" (
  "client_id" integer NOT NULL
,  "gender" varchar(1) NOT NULL
,  "birth_date" date NOT NULL
,  "district_id" integer NOT NULL
,  PRIMARY KEY ("client_id")
,  CONSTRAINT "client_ibfk_1" FOREIGN KEY ("district_id") REFERENCES "district" ("district_id")
);
CREATE TABLE "disp" (
  "disp_id" integer NOT NULL
,  "client_id" integer NOT NULL
,  "account_id" integer NOT NULL
,  "type" varchar(9) NOT NULL
,  PRIMARY KEY ("disp_id")
,  CONSTRAINT "disp_ibfk_1" FOREIGN KEY ("account_id") REFERENCES "account" ("account_id")
,  CONSTRAINT "disp_ibfk_2" FOREIGN KEY ("client_id") REFERENCES "client" ("client_id")
);
CREATE TABLE "district" (
  "district_id" integer NOT NULL DEFAULT '0'
,  "A2" varchar(19) NOT NULL
,  "A3" varchar(15) NOT NULL
,  "A4" integer NOT NULL
,  "A5" integer NOT NULL
,  "A6" integer NOT NULL
,  "A7" integer NOT NULL
,  "A8" integer NOT NULL
,  "A9" integer NOT NULL
,  "A10" decimal(4,1) NOT NULL
,  "A11" integer NOT NULL
,  "A12" decimal(4,1) DEFAULT NULL
,  "A13" decimal(3,2) NOT NULL
,  "A14" integer NOT NULL
,  "A15" integer DEFAULT NULL
,  "A16" integer NOT NULL
,  PRIMARY KEY ("district_id")
);
CREATE TABLE "loan" (
  "loan_id" integer NOT NULL DEFAULT '0'
,  "account_id" integer NOT NULL
,  "date" date NOT NULL
,  "amount" integer NOT NULL
,  "duration" integer NOT NULL
,  "payments" decimal(6,2) NOT NULL
,  "status" varchar(1) NOT NULL
,  PRIMARY KEY ("loan_id")
,  CONSTRAINT "loan_ibfk_1" FOREIGN KEY ("account_id") REFERENCES "account" ("account_id")
);
CREATE TABLE "order" (
  "order_id" integer NOT NULL DEFAULT '0'
,  "account_id" integer NOT NULL
,  "bank_to" varchar(2) NOT NULL
,  "account_to" integer NOT NULL
,  "amount" decimal(6,1) NOT NULL
,  "k_symbol" varchar(8) NOT NULL
,  PRIMARY KEY ("order_id")
,  CONSTRAINT "order_ibfk_1" FOREIGN KEY ("account_id") REFERENCES "account" ("account_id")
);
CREATE TABLE "trans" (
  "trans_id" integer NOT NULL DEFAULT '0'
,  "account_id" integer NOT NULL DEFAULT '0'
,  "date" date NOT NULL
,  "type" varchar(6) NOT NULL
,  "operation" varchar(14) DEFAULT NULL
,  "amount" integer NOT NULL
,  "balance" integer NOT NULL
,  "k_symbol" varchar(11) DEFAULT NULL
,  "bank" varchar(2) DEFAULT NULL
,  "account" integer  DEFAULT NULL
,  PRIMARY KEY ("trans_id")
,  CONSTRAINT "trans_ibfk_1" FOREIGN KEY ("account_id") REFERENCES "account" ("account_id")
);"""  # noqa: E501

    """CzechBankQA"""
    name = "czech_bank_qa"
    description = "This is a list of SQL queries for a text-to-SQL task over the Czech Bank 1999 dataset."
    tags = ["text_to_sql"]

    def __init__(self, config_name: str):
        super().__init__()
        self.config_name = config_name

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "yifanmai/czech_bank_qa", name=self.config_name, split="test", cache_dir=cache_dir
        )
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["description"])
            references = [Reference(output=Output(text=row["sql_query"]), tags=[CORRECT_TAG])]
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances
