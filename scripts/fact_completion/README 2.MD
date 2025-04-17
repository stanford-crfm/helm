# Fact Completion Dataset Generation Process

## Task overview

The fact completition dataset tests the relational knowledge contained in a model. We identified a set of Wikidata relations (referred to as our `seed relations`) spanning different subject matters (e.g. law, philosophy, computer science, and etc.). For each relation, we sampled triples from Wikidata containing that relation, designed a template sentence which encodes the fact in English, and generated sentences according to this template for each triple. For example, for the relation [P19](https://www.wikidata.org/wiki/Property:P19), corresponding to "place of birth," our sample could include the triple ([Q11975](https://www.wikidata.org/wiki/Q11975), P19, [Q846178](https://www.wikidata.org/wiki/Q846178)), which corresponds to Britney Spears and McComb. Our template for P19 is "[X] was born in", wheren [X] corresponds to the *head* of the triple (in this case, Q11975). We sample a random alias (from the list Wikidata associates with each entity) for the head entity to use in [X]. Thus, our sentence here might be "Britney Spears was born in". The model is prompted to complete the sentence, and is judged to be correct if the generated text corresponds to a Wikidata alias for the tail entity (which here, would be Q846178).  

## Dataset construction

We provide code for replicating our data generation process.

### Relations

CSV files listing the seed relations can be downloaded from the [CRFM website](https://nlp.stanford.edu/crfm/benchmarking/data/wikidata_relations.zip). Each TSV file contains relations for different domains.

### Wikidata processing

Given the size of Wikidata, we use [simple-wikidata-db](https://github.com/neelguha/simple-wikidata-db) to preprocess the raw dump and extract triples. We used the Wikidata dump from January 2022. Roughly, the simple-wikidata-db library creates tables to store different types of triples, where a table consists of a directory with JSONL files (where a line of a single file corresponds to a triple, with keys for the head, relation, and tail). Saving the data in this structure has two advantages: 

1. If we know we want triples containing a certain type of information, we only have to load the tables which contain those triples. 
2. Distributing the contents of a table across multiple files within a directory allows us to easily process files in parallel. 

The simple-wikidata-db repository contains information on the full set of tables it creates. For our purposes, we care about the following tables: 

- `entity_rels`: which holds triples where both the head and tail are Wikidata entities.
- `entity_vals`: which holds triples where the head is an entity, but the tail is a value (e.g. a string, float, integer, etc.).
- `aliases`: which holds alias information for each entity, i.e. the set of names/aliases an entity is associated with.

### Generating samples

First, run `fetch_triples_and_aliases.py` to (1) extract all triples corresponding to the seed relations, and (2) the aliases associated with QIDs found in these triples.

```console
> python3 scripts/fact_completion/fetch_triples_and_aliases.py --processed_wikidata $path_to_folder_with_processed_wikidata_dump
```

Next, we filter the triples. In particular we remove triples where: 

- Either the head or tail entity do not have a Wikipedia page. 
- The entity corresponds to a category, template, stub, disamibguation, or list page.

```console
> python3 scripts/completition/filter_triples.py --processed_wikidata $path_to_folder_with_processed_wikidata_dump
```

Finally, we sample triples to include as part of the benchmark:

```console
> python3 create_benchmark.py
```
