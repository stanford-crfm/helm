"""
This script fetches all QIDs which participate in a predefined list of relations. It writes two files: 

    1) triples.tsv: the triples with relations in the wikidata benchmark. 
    2) names.tsv: the list of aliases for each qid mentioned in the benchmark. 
"""
import argparse
from functools import partial
import glob
from multiprocessing import Pool
import os
from tqdm import tqdm

from utils import *


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_wikidata", 
        type=str, 
        help="path to processed wikidata dump (see simple-wikidata-db)."
    )
    parser.add_argument(
        "--num_procs", 
        type=int, 
        default=10, 
        help="Number of processes"
    )
    parser.add_argument(
        "--benchmark_folder",
        type=str,
        help="directory to write benchmark data to"
    )
    return parser


def rel_filtering_func(rels, filename):
    filtered = []
    for item in jsonl_generator(filename):
        if item["property_id"] in rels:
            filtered.append(item)
    return filtered

def alias_filtering_func(qids, filename):
    filtered = []
    for item in jsonl_generator(filename):
        if item["qid"] in qids:
            filtered.append(item)
    return filtered

def load_relations(fpath):
    rels = []
    with open(fpath) as in_file: 
        for line in in_file: 
            rels.append(line.strip().split(",")[0])
    print(f"Loaded {len(rels)} from {fpath}.")
    return rels

def main():
    args = get_arg_parser().parse_args()

    # load benchmark relations 
    fpaths = glob.glob(os.path.join(args.relations_folder, "*.csv"))
    benchmark_relations = set()
    for fpath in fpaths: 
        benchmark_relations.update(load_relations(fpath))
    
    # Load table files
    table_files = get_batch_files(os.path.join(args.processed_wikidata, "entity_rels"))
    table_files.extend(get_batch_files(os.path.join(args.processed_wikidata, "entity_values")))

    # Run on only a subset of the data (useful for debugging)
    if args.test:
        table_files = table_files[:1]

    pool = Pool(processes=args.num_procs)
    filtered = []
    for output in tqdm(
            pool.imap_unordered(
                partial(rel_filtering_func, benchmark_relations), table_files, chunksize=1),
            total=len(table_files)
    ):
        filtered.extend(output)

    print(f"Extracted {len(filtered)} triples with benchmark relations.")

    # Write all triples to file
    triples_file = os.path.join(args.benchmark_folder, "triples.tsv")
    with open(triples_file, "w") as out_file:
        for item in tqdm(filtered):
            qid, property_id, val = item["qid"], item["property_id"], item["value"]
            out_file.write(f"{qid}\t{property_id}\t{val}\n")


    # We also extract the aliases for each qid  
    qids = set()
    for item in filtered:
        qids.add(item["qid"])
        qids.add(item["value"])
    
    table_files = get_batch_files(os.path.join(args.processed_wikidata, "aliases"))
    pool = Pool(processes=args.num_procs)
    filtered = []
    for output in tqdm(
            pool.imap_unordered(
                partial(alias_filtering_func, qids), table_files, chunksize=1),
            total=len(table_files)
    ):
        filtered.extend(output)

    print(f"Extracted {len(filtered)} names for QIDS with benchmark relations.")
    names_file = os.path.join(args.benchmark_folder, "names.tsv")
    with open(names_file, "w") as out_file:
        for item in tqdm(filtered):
            qid, alias = item["qid"], item["alias"]
            out_file.write(f"{qid}\t{alias}\n")

    


if __name__ == "__main__":
    main()