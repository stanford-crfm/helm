#!/bin/bash

#run pre commit
pip install --upgrade pip
pip install -r requirements-freeze.txt
pip install -e .

# Run benchmark
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite v1

# Start a web server to display benchmark results
helm-server
