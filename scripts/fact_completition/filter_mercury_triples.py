"""
This script filters out all triples which have "bad" names and don't have wikipedia pages. 

"""
import argparse
from collections import defaultdict
import json
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
        "--test", 
        action="store_true",
        help="Runs on only a subset of the data (used to test pipeline)."
    )
    parser.add_argument(
        "--relations_folder",
        type=str,
        help="path to folder with benchmark relations CSVs."
    )
    parser.add_argument(
        "--benchmark_folder",
        type=str,
        help="directory to write data to"
    )
    return parser


def alias_filtering_func(filename):
    filtered = []
    for item in jsonl_generator(filename):
        filtered.append(item)
    return filtered

def bad_alias(aliases):
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


def main():
    args = get_arg_parser().parse_args()

    # Get names for qids and build alias map
    table_files = get_batch_files(os.path.join(args.processed_wikidata, "aliases"))
    qid2alias = defaultdict(list)
    for filename in tqdm(table_files):
        for item in jsonl_generator(filename):
            qid = item['qid']
            alias = item['alias']
            qid2alias[qid].append(alias)
    print(f"Built alias map for {len(qid2alias)} qids.")


    # wikipedia page set
    wikipedia_ids = set() # qids with wikipedia pages
    table_files = get_batch_files(os.path.join(args.processed_wikidata, "wikipedia_links"))
    for f in tqdm(table_files):
        with open(f) as in_file:
            for line in in_file: 
                item = json.loads(line)
                wikipedia_ids.add(item['qid'])


    # load mercury qids 
    triples_file = os.path.join(args.benchmark_folder, "triples.tsv")
    triples = []
    with open(triples_file, "r") as in_file: 
        for line in tqdm(in_file, total=215288006): 
            q1, p, q2 = line.strip().split("\t")
            triples.append((q1, p, q2))
    print(f"Loaded {len(triples)} triples.")

    # filter out triples corresponding to "bad" aliases 
    filtered_triples = []
    for triple in tqdm(triples):
        q1, p, q2 = triple
        if bad_alias(qid2alias[q1]) or bad_alias(qid2alias[q2]):
            continue 
        if not q1 in wikipedia_ids:
            continue 
        if not q2 in wikipedia_ids:
            continue 
        filtered_triples.append([q1, p, q2])
    
    print(f"{len(filtered_triples)} triples after filtering.")

    # save to file
    filtered_triples_file = os.path.join(args.benchmark_folder, "filtered_triples.tsv")
    with open(filtered_triples_file, "w") as out_file:
        for item in tqdm(filtered_triples):
            qid, property_id, val = item
            out_file.write(f"{qid}\t{property_id}\t{val}\n")

if __name__ == "__main__":
    main()