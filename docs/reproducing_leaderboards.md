# Reproducing Leaderboards

You can use the HELM package to rerun evaluation runs and recreate a specific public leaderboard.

The general procedure is to first find the appropriate `run_entries_*.conf` and `schema_*.yaml` files from the HELM GitHub repository for the leaderboard version and then place them in your current working directory. The locations of these files are as follows:

- `run_entries_*.conf`: the `src/helm/benchmark/presentation/` directory [here](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation)
- `schema_*.conf`: the `src/helm/benchmark/static/` directory [here](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/static)

Then run the following shell script:

```bash
# Pick any suite name of your choice
export SUITE_NAME=my_suite

# Replace this with your model or models
export MODELS_TO_RUN=openai/gpt-3.5-turbo-0613

# Get these from the list below
export RUN_ENTRIES_CONF_PATH=run_entries_repro.conf
export SCHEMA_PATH=schema_repro.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2

helm-run --conf-paths $RUN_ENTRIES_CONF_PATH --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --priority $PRIORITY --suite $SUITE_NAME --models-to-run $MODELS_TO_RUN

helm-summarize --schema $SCHEMA_PATH --suite $SUITE_NAME

helm-server --suite $SUITE_NAME
```

## Leaderboard versions

The following specifies the appropriate parameters and configuration files for a leaderboard, given its project and version number.

### Lite

```bash
export RUN_ENTRIES_CONF_PATH=run_entries_lite_20240424.conf
export SCHEMA_PATH=schema_lite.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2
```

### Classic before v0.2.4

```bash
export RUN_ENTRIES_CONF_PATH=run_entries.conf
export SCHEMA_PATH=schema_classic.yaml
export NUM_TRAIN_TRIALS=3
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2
```

### Classic v0.2.4 and after

```bash
export RUN_ENTRIES_CONF_PATH=run_entries_lite.conf
export SCHEMA_PATH=schema_classic.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2
```

### HEIM

```bash
export RUN_ENTRIES_CONF_PATH=run_entries_heim.conf
export SCHEMA_PATH=schema_heim.yaml
export NUM_TRAIN_TRIALS=1
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2
```

### MMLU

```bash
export RUN_ENTRIES_CONF_PATH=run_entries_mmlu.conf
export SCHEMA_PATH=schema_mmlu.yaml
export NUM_TRAIN_TRIALS=1
export NUM_EVAL_INSTANCES=10000
export PRIORITY=4
```

### VHELM v1.0.0 (VLM evaluation)

```bash
export RUN_ENTRIES_CONF_PATH=run_entries_vhelm_lite.conf
export SCHEMA_PATH=schema_vhelm_lite.yaml
export NUM_TRAIN_TRIALS=1
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2
```