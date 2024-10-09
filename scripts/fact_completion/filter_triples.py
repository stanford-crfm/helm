"""
This script filters out triples where either the head or tail entity

    - do not have an English Wikipedia page, or
    - correspond to a category, template, stub, disambiguation, or list
"""

import argparse
from collections import defaultdict
import json
from tqdm import tqdm
from pathlib import Path
from utils import get_batch_files, jsonl_generator, save_jsonl


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_wikidata", type=str, help="Path to processed wikidata dump (see simple-wikidata-db)."
    )
    parser.add_argument(
        "--benchmark_folder", type=str, default="./benchmark", help="Directory to write benchmark data to."
    )
    parser.add_argument(
        "--wikipedia_links_folder", type=str, default="wikipedia_links", help="Folder containing Wikipedia links table."
    )
    return parser


def bad_alias(aliases: list) -> bool:
    """Returns true if an entity has a "bad" alias and false otherwise. An alias is bad
    if it corresponds to a category, stub, disambiguation, stub, template, or list
    page. We just check for these keywords in the title.

    Args:
        aliases (list): list of aliases for an entity.

    Returns:
        bool: whether or not the entity has a bad alias.
    """
    if len(aliases) == 0:
        return True
    for a in aliases:
        if "category" in a.lower():
            return True
        if "stub" in a.lower():
            return True
        if "disambiguation" in a.lower():
            return True
        if "template" in a.lower():
            return True
        if "list of" in a.lower():
            return True
    return False


def main() -> None:
    args = get_arg_parser().parse_args()

    processed_folder = Path(args.processed_wikidata)
    benchmark_folder = Path(args.benchmark_folder)

    # Get qid2alias mapping
    aliases_fpath = benchmark_folder / "aliases.jsonl"
    print(f"Loading aliases from {aliases_fpath}.")
    qid2alias = defaultdict(list)
    for item in jsonl_generator(str(aliases_fpath)):
        qid = item["qid"]
        alias = item["alias"]
        qid2alias[qid].append(alias)
    print(f"Built alias map for {len(qid2alias)} qids.")

    # Set of QIDs with Wikipedia pages
    print("Loading Wikipedia table.")
    wikipedia_qids = set()
    table_files = get_batch_files(processed_folder / args.wikipedia_links_folder)
    for f in tqdm(table_files, disable=None):
        with open(f) as in_file:
            for line in in_file:
                item = json.loads(line)
                wikipedia_qids.add(item["qid"])
    print(f"Found {len(wikipedia_qids)} QIDs with Wikpedia pages.")

    # load triples with seed relations
    triples_file = benchmark_folder / "triples.jsonl"
    print(f"Loading triples from {triples_file}. Filtering based on alias quality...")
    filtered_triples = []
    for triple in tqdm(jsonl_generator(str(triples_file)), total=215288006, disable=None):
        q1 = triple["qid"]
        q2 = triple["value"]
        if bad_alias(qid2alias[q1]) or bad_alias(qid2alias[q2]):
            continue
        if q1 not in wikipedia_qids:
            continue
        if q2 not in wikipedia_qids:
            continue
        filtered_triples.append(triple)

    print(f"{len(filtered_triples)} triples after filtering.")

    # save to file
    filtered_triples_file = benchmark_folder / "filtered_triples.jsonl"
    save_jsonl(filtered_triples_file, filtered_triples)
    print(f"Saved filtered triples to {filtered_triples_file}.")


if __name__ == "__main__":
    main()
