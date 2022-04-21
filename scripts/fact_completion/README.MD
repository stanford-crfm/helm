# Fact Completion Dataset Generation Process

## Task overview

The fact completition dataset tests the relational knowledge contained in a model. We identified a set of Wikidata relations (referred to as our `seed relations`) spanning different subject matters (e.g. law, philosophy, computer science, and etc.). For each relation, we sampled triples from Wikidata containing that relation, designed a template sentence which encodes the fact in English, and generated sentences according to this template for each triple. For example, for the relation [P19](https://www.wikidata.org/wiki/Property:P19), corresponding to "place of birth," our sample could include the triple ([Q11975](https://www.wikidata.org/wiki/Q11975), P19, [Q846178](https://www.wikidata.org/wiki/Q846178)), which corresponds to Britney Spears and McComb. Our template for P19 is "[X] was born in", wheren [X] corresponds to the *head* of the triple (in this case, Q11975). We sample a random alias (from the list Wikidata associates with each entity) for the head entity to use in [X]. Thus, our sentence here might be "Britney Spears was born in". The model is prompted to complete the sentence, and is judged to be correct if the generated text corresponds to a Wikidata alias for the tail entity (which here, would be Q846178).  

## Dataset construction

We provide code for replicating our data generation process.

### Relations

CSV files listing the seed relations can be downloaded from the [CRFM website](https://nlp.stanford.edu/crfm/benchmarking/data/wikidata_relations.zip). Each TSV file contains relations for different domains.

### Wikidata processing

Given the size of Wikidata, we use [simple-wikidata-db](https://github.com/neelguha/simple-wikidata-db) to preprocess the raw dump and extract triples. We used the Wikidata dump from January 2022.

### Generating samples

First, run `fetch_triples_and_aliases.py` to (1) extract all triples corresponding to the seed relations, and (2) the aliases associated with QIDs found in these triples.

```console
> python3 fetch_triples_and_aliases.py --processed_wikidata $path_to_folder_with_processed_wikidata_dump
```


This folder contains code to generate the Wikidata fact completition dataset. 

Scripts:

- `fetch_mercury_relations.py`: collects triples corresponding to relations included in the benchmark. Also collects aliases for all QIDS which participate in these relations.
- `filter_mercury_triples.py`: filters triples to exclude those which (1) do not have a wikipedia page, and (2) are "bad" entities (e.g. list pages, stubs, disambiguation pages, etc.)
- `create_mercury_dataset.py`: samples triples to include in benchmark, and creates template prompts.
