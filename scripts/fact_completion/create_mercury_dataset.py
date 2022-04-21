"""
This script creates the dataset for the mercury relations.

1. Subsample 1k triples per relation.
2. Get names of QIDs for each and construct sentences.

"""
import argparse
from collections import defaultdict
import glob
import json
import numpy as np
import os
from tqdm import tqdm

from utils import jsonl_generator


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_wikidata", type=str, help="path to processed wikidata dump (see simple-wikidata-db)."
    )
    parser.add_argument("--num_procs", type=int, default=10, help="Number of processes")
    parser.add_argument(
        "--test", action="store_true", help="Runs on only a subset of the data (used to test pipeline)."
    )
    parser.add_argument("--relations_folder", type=str, help="path to folder with benchmark relations CSVs.")
    parser.add_argument("--benchmark_folder", type=str, help="directory to write data to")
    return parser


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

    # load templates containing relations + sentences
    fpaths = glob.glob(os.path.join(args.relations_folder, "*.csv"))
    relation_to_domain_map = {}
    templates = {}
    for fpath in fpaths:
        domain = fpath.split("/")[-1]
        domain = domain.replace(".csv", "")
        domain = domain.replace("wikidata relations - ", "")
        with open(fpath) as in_file:
            for line in in_file:
                items = line.strip().split(",")
                relation = items[0]
                template = items[1]
                templates[relation] = template
                relation_to_domain_map[relation] = domain

    print(f"Loaded {len(templates)} templates.")

    # load all triples in the form of a prop_map, which maps
    # property -> head_qid -> tail_qid
    filtered_triples_file = os.path.join(args.benchmark_folder, "filtered_triples.tsv")
    prop_map = defaultdict(lambda: defaultdict(list))
    with open(filtered_triples_file, "r") as in_file:
        for line in tqdm(in_file, total=182056518):
            q1, p, q2 = line.strip().split("\t")
            prop_map[p][q1].append(q2)
    print(f"Loaded prop map for {len(prop_map)} items.")

    # Select head QIDS to include in dataset
    selected = []
    all_qids = set()
    for p in tqdm(prop_map):
        np.random.seed(0)
        head_qids = list(prop_map[p].keys())
        idxs = np.random.choice(len(head_qids), min(1000, len(head_qids)), replace=False)
        for i in idxs:
            head_qid = head_qids[i]
            all_qids.add(head_qid)
            all_qids.update(prop_map[p][head_qid])
            selected.append([head_qid, p])
    print(f"{len(selected)} total triples ({len(all_qids)} total qids).")

    # Load QID names
    qid_names = defaultdict(list)
    names_file = os.path.join(args.benchmark_folder, "names.tsv")
    with open(names_file, "w") as in_file:
        for line in in_file:
            item = json.loads(line)
            qid, alias = item["qid"], item["alias"]
            qid_names[qid].append(alias)

    # Create dataset with templates
    dataset = []
    prop_counts = defaultdict(int)
    for head_qid, prop in selected:
        prop_counts[prop] += 1
        if len(qid_names[head_qid]) == 0:
            continue

        template = templates[prop]
        query_name = qid_names[head_qid][0]
        template = template.replace("[X]", query_name)
        template = template.strip()

        result_qids = [q for q in prop_map[prop][head_qid] if len(qid_names[q]) > 0]
        result_names = [qid_names[q] for q in result_qids]
        if len(result_names) == 0:
            continue
        output = {
            "template": template,
            "query_qid": head_qid,
            "query_name": query_name,
            "result_qids": result_qids,
            "result_names": result_names,
            "property": prop,
            "domain": relation_to_domain_map[prop],
        }
        dataset.append(output)

    # split to train/dev/test
    train_count, dev_count = {}, {}
    for prop, count in prop_counts.items():
        if count < 10:
            train_count[prop] = 1
            dev_count[prop] = 0
        elif count < 100:
            train_count[prop] = 10
            dev_count[prop] = 5
        elif count < 200:
            train_count[prop] = 30
            dev_count[prop] = 15
        else:
            train_count[prop] = 100
            dev_count[prop] = 50

    train = defaultdict(list)
    dev = defaultdict(list)
    test = defaultdict(list)
    for item in dataset:
        prop = item["property"]
        if len(train[prop]) < train_count[prop]:
            train[prop].append(item)
        elif len(dev[prop]) < dev_count[prop]:
            dev[prop].append(item)
        else:
            test[prop].append(item)

    # save splits
    datasets = [train, dev, test]
    names = ["train", "dev", "test"]
    for split, name in zip(datasets, names):
        fpath = os.path.join(args.benchmark_folder, f"{name}.jsonl")
        with open(fpath, "w") as out_file:
            for items in split.values():
                for item in items:
                    out_file.write(json.dumps(item) + "\n")

    # save all data
    fpath = os.path.join(args.benchmark_folder, "all_data.jsonl")
    with open(fpath, "w") as out_file:
        for item in dataset:
            out_file.write(json.dumps(item) + "\n")

    # Save stats
    fpath = os.path.join(args.benchmark_folder, "stats.json")
    with open(fpath, "w") as out_file:
        out_file.write(json.dumps(dict(prop_counts)))


if __name__ == "__main__":
    main()
