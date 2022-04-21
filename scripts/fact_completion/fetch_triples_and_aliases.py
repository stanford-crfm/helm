"""
This script filters the wikidata dump to only triples corresponding to identified relations. It writes two files:

    1) triples.tsv: containing triples corresponding to the seed relations.
    2) names.tsv: aliases for each QID found in triples.tsv.
"""
import argparse
from functools import partial
import glob
from multiprocessing import Pool
import os
from tqdm import tqdm
from typing import Set, List, Dict
from pathlib import Path
from utils import get_batch_files, jsonl_generator, load_seed_relations


def get_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_wikidata", type=str, help="Path to processed wikidata dump (see simple-wikidata-db)."
    )
    parser.add_argument("--num_procs", type=int, default=10, help="Number of processes.")
    parser.add_argument("--benchmark_folder", type=str, help="Directory to write benchmark data to.")
    parser.add_argument("--relations_folder", type=str, help="Folder containing tsv files for seed relations.")
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

    # Load seed relations
    seed_df = load_seed_relations(relations_folder)
    seed_relations = set(seed_df["relation"])

    # Load processed wikidata files corresponding to entity relations and entity values.
    # Here, we assume the data dump is structured as in https://github.com/neelguha/simple-wikidata-db
    table_files = get_batch_files(processed_folder / "entity_rels")
    table_files.extend(get_batch_files(processed_folder / "entity_values"))

    # Extract triples containing a seed relation
    pool = Pool(processes=args.num_procs)
    filtered = []
    for output in tqdm(
        pool.imap_unordered(partial(rel_filtering_func, seed_relations), table_files, chunksize=1),
        total=len(table_files),
    ):
        filtered.extend(output)

    print(f"Extracted {len(filtered)} triples for seed relations.")

    # Save triples to file.
    # TODO: this should probably be in JSONL format?
    triples_file = os.path.join(args.benchmark_folder, "triples.tsv")
    with open(triples_file, "w") as out_file:
        for item in tqdm(filtered):
            qid, property_id, val = item["qid"], item["property_id"], item["value"]
            out_file.write(f"{qid}\t{property_id}\t{val}\n")

    # Identity list of QIDs mentioned in extracted triples.
    qids = set()
    for item in filtered:
        qids.add(item["qid"])
        qids.add(item["value"])

    # Load tables corresponding to aliases
    table_files = get_batch_files(processed_folder / "aliases")
    pool = Pool(processes=args.num_procs)
    filtered = []
    for output in tqdm(
        pool.imap_unordered(partial(alias_filtering_func, qids), table_files, chunksize=1), total=len(table_files)
    ):
        filtered.extend(output)
    print(f"Extracted {len(filtered)} aliases.")

    # Save to file.
    names_file = os.path.join(args.benchmark_folder, "names.tsv")
    with open(names_file, "w") as out_file:
        for item in tqdm(filtered):
            qid, alias = item["qid"], item["alias"]
            out_file.write(f"{qid}\t{alias}\n")


if __name__ == "__main__":
    main()
