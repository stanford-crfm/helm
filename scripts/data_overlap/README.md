# Data Overlap Script (no dependencies on HELM)


## Installation

```bash
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3 venv

# Activate the virtual environment.
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running

```bash

This needs to be run from the data overlap directory; i.e. cd scripts/data_overlap if you are at the top level HELM folder

Usage:

python [compute_data_overlap_metrics.py OR run_data_overlap_beam.py] --input-data <input_data> --scenario-data <scenario_data> --output-stats <output_stats> --input-format <input_format>

For instance, you can call this with The Pile, e.g. have:
    input_data  = 00.jsonl (download https://pile.eleuther.ai/)
    scenario_data = (example included with repo, but can use HELM to generate)
    output_stats = arbitrary output file name, e.g. "output_stats"
    input_format = the_pile

If you don't want to output the ngrams that are overlapping in test set to a separate "{output_stats}_ngrams" file, you can pass --no-output-ngrams.

There are additional optional args:
--normalization default 
--tags tag1 tag2
```

## Docker

To create and run docker image:

    docker build . -t data_overlap_script
    docker run --input-path=<input-path> --scenario-data=<scenario-data> --output-stats=<output-stats> --input-format=<input-format> --rm -it data_overlap_script:latest 

example with values:
    
    docker run --rm -it data_overlap_script:latest --input-path="input.json" --scenario-data="scenario_data" --output-stats="output_stats" --input-format="the_pile"

You'll need some way of providing access to your files for the computation, such as [bind mounts](https://docs.docker.com/storage/bind-mounts/) or [volumes](https://docs.docker.com/storage/volumes/)
