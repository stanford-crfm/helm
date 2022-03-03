# Fact Completion Dataset Generation Process 


This folder contains code to generate the Wikidata fact completition dataset. We use [simple-wikidata-db](https://github.com/neelguha/simple-wikidata-db) to preprocess and extract triples from Wikidata. We used the Wikidata dump from January 2022. 


Scripts: 
- `fetch_merucy_relations.py`: collects triples corresponding to relations included in the benchmark. Also collects aliases for all QIDS which participate in these relations. 
- `filter_mercury_triples.py`: filters triples to exclude those which (1) do not have a wikipedia page, and (2) are "bad" entities (e.g. list pages, stubs, disambiguation pages, etc.)
- `create_mercury_dataset.py`: samples triples to include in benchmark, and creates template prompts. 
