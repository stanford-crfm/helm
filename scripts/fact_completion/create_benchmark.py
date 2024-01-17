# mypy: check_untyped_defs = False
"""
This script samples triples to use and constructs sentences from templates.

1. Subsample 1k triples per relation.
2. Get names of QIDs for each and construct sentences.

"""
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import jsonl_generator, load_seed_relations, save_jsonl


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_folder", type=str, default="./benchmark", help="Directory to write benchmark data to."
    )
    parser.add_argument(
        "--relations_folder",
        type=str,
        default="./wikidata_relations",
        help="Folder containing tsv files for seed relations.",
    )
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

    relations_folder = Path(args.relations_folder)
    benchmark_folder = Path(args.benchmark_folder)

    # load templates containing relations + sentences
    seed_relations_df = load_seed_relations(relations_folder)
    relation_to_domain_map = {}
    templates = {}
    for i in range(len(seed_relations_df)):
        domain = seed_relations_df.iloc[i]["domain"]
        relation = seed_relations_df.iloc[i]["relation"]
        template = seed_relations_df.iloc[i]["template"]
        templates[relation] = template
        relation_to_domain_map[relation] = domain
    print(f"Loaded {len(templates)} templates.")

    # load all triples in the form of a prop_map, which maps
    # property -> head_qid -> tail_qid
    filtered_triples_file = benchmark_folder / "filtered_triples.jsonl"
    prop_map = defaultdict(lambda: defaultdict(list))
    for item in jsonl_generator(filtered_triples_file):
        head = item["qid"]
        property = item["property_id"]
        tail = item["value"]
        prop_map[property][head].append(tail)
    print(f"Loaded prop map for {len(prop_map)} items.")

    # Select head QIDS to include in dataset
    selected = []
    all_qids = set()
    for p in tqdm(prop_map, disable=None):
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
    aliases_file = benchmark_folder / "aliases.jsonl"
    for item in jsonl_generator(aliases_file):
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
        fpath = benchmark_folder / f"{name}.jsonl"
        samples = [item for items in split.values() for item in items]
        save_jsonl(fpath, samples)

    # save all data
    fpath = benchmark_folder / "all_data.jsonl"
    save_jsonl(fpath, dataset)

    # Save stats
    fpath = benchmark_folder / "stats.json"
    save_jsonl(fpath, [dict(prop_counts)])


if __name__ == "__main__":
    main()
