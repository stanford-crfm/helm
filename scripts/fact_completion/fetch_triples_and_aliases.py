"""
This script filters the wikidata dump to only triples corresponding to identified relations. It writes two files:

    1) triples.tsv: containing triples corresponding to the seed relations.
    2) names.tsv: aliases for each QID found in triples.tsv.
"""

import argparse
from tqdm import tqdm
from typing import Set, List, Dict
from pathlib import Path
from utils import get_batch_files, jsonl_generator, load_seed_relations, save_jsonl


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_wikidata", type=str, help="Path to processed wikidata dump (see simple-wikidata-db)."
    )
    parser.add_argument("--num_procs", type=int, default=10, help="Number of processes.")
    parser.add_argument(
        "--benchmark_folder", type=str, default="./benchmark", help="Directory to write benchmark data to."
    )
    parser.add_argument(
        "--relations_folder",
        type=str,
        default="./wikidata_relations",
        help="Folder containing tsv files for seed relations.",
    )
    parser.add_argument(
        "--entity_rels_folder", type=str, default="entity_rels", help="Folder containing entity relations table."
    )
    parser.add_argument(
        "--entity_values_folder", type=str, default="entity_values", help="Folder containing entity values table."
    )
    parser.add_argument("--aliases_folder", type=str, default="aliases", help="Folder containing entity aliases table.")
    return parser


def rel_filtering_func(seed_rels: Set[str], filepath: str) -> List[Dict[str, str]]:
    """Returns triples with relation in seed_rels

    Args:
        seed_rels (Set[str]): filtering set of relations.
        filepath (str): path to jsonl file with triple data

    Returns:
        List[Dict[str, str]]: list of filtered triples.
    """
    filtered = []
    for item in jsonl_generator(filepath):
        if item["property_id"] in seed_rels:
            filtered.append(item)
    return filtered


def alias_filtering_func(qids: Set[str], filename: str) -> List[Dict[str, str]]:
    filtered = []
    for item in jsonl_generator(filename):
        if item["qid"] in qids:
            filtered.append(item)
    return filtered


def main() -> None:
    args = get_arg_parser().parse_args()

    relations_folder = Path(args.relations_folder)
    processed_folder = Path(args.processed_wikidata)
    benchmark_folder = Path(args.benchmark_folder)
    benchmark_folder.mkdir(exist_ok=True)

    # Load seed relations
    seed_df = load_seed_relations(relations_folder)
    seed_relations = set(seed_df["relation"])
    print(f"Loaded {len(seed_relations)} seed relations.")

    # Load processed wikidata files corresponding to entity relations and entity values.
    # Here, we assume the data dump is structured as in https://github.com/neelguha/simple-wikidata-db
    table_files = get_batch_files(processed_folder / args.entity_rels_folder)
    table_files.extend(get_batch_files(processed_folder / args.entity_values_folder))

    filtered_triples = []
    for filepath in tqdm(table_files, disable=None):
        filtered_triples.extend(rel_filtering_func(seed_relations, filepath))
    print(f"Extracted {len(filtered_triples)} triples for seed relations.")

    triples_file = benchmark_folder / "triples.jsonl"
    save_jsonl(triples_file, filtered_triples)
    print(f"Written triples to {triples_file}.")

    # Extract and filter aliases for QIDs mentioned in the above triples.
    qids = set()
    for item in filtered_triples:
        qids.add(item["qid"])
        qids.add(item["value"])

    table_files = get_batch_files(processed_folder / args.aliases_folder)
    filtered_aliases = []
    for filepath in tqdm(table_files, disable=None):
        filtered_aliases.extend(alias_filtering_func(qids, filepath))
    print(f"Extracted {len(filtered_aliases)} aliases.")

    # Save aliases to file.
    aliases_file = benchmark_folder / "aliases.jsonl"
    save_jsonl(aliases_file, filtered_aliases)
    print(f"Written aliases to {aliases_file}.")


if __name__ == "__main__":
    main()
