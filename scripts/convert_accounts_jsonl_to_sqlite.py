import argparse
import json
from dataclasses import asdict

from dacite import from_dict

from proxy.accounts import Account
from sqlitedict import SqliteDict


"""
Script to convert existing accounts.jsonl to accounts.sqlite.

Usage:

    python3 scripts/convert_accounts_jsonl_to_sqlite.py prod_env/accounts.jsonl -o prod_env/accounts.sqlite

"""


def convert_accounts_jsonl_to_sqlite(jsonl_path: str, output_path: str):
    with SqliteDict(output_path) as cache:
        for line in open(jsonl_path):
            account: Account = from_dict(data_class=Account, data=json.loads(line))
            cache[account.api_key] = asdict(account)
        cache.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", type=str, help="Path to accounts.jsonl")
    parser.add_argument(
        "-o", "--output-path", type=str, default="prod_env/accounts.sqlite", help="Path to AI21 cache.",
    )
    args = parser.parse_args()

    convert_accounts_jsonl_to_sqlite(args.jsonl_path, args.output_path)
    print("Done.")
