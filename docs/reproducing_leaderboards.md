# Reproducing Leaderboards

You can use the HELM package to rerun evaluation runs and recreate a specific public leaderboard.

The general procedure is to first find the appropriate `run_entries_*.conf` and `schema_*.yaml` files from the HELM GitHub repository for the leaderboard version, and then place them in your current working directory. The locations of these files are as follows:

- `run_entries_*.conf`: the `src/helm/benchmark/presentation/` directory
- `schema_*.conf`: the `src/helm/benchmark/static/` directory

Then run the following shell script:

```bash
# Pick any suite name of your choice
export SUITE_NAME=repro

# Get these from the list below
export RUN_ENTRIES_CONF_PATH=run_entries_repro.conf
export SCHEMA_PATH=schema_repro.yaml
export NUM_TRAIN_TRIALS=3
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2

helm-run --conf-files $RUN_ENTRIES_CONF_PATH --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --priority $PRIORITY --suite $SUITE_NAME
helm-summarize --schema $SCHEMA_PATH --suite $SUITE_NAME
helm-server --suite $SUITE_NAME
```

## Leaderboard versions

The following specifies the appropriate parameters and configuration files for a leaderboard given its project and version number.

#### Classic before v0.2.4

```bash
export RUN_ENTRIES_CONF_PATH=run_specs.conf
export SCHEMA_PATH=schema_classic.yaml
export NUM_TRAIN_TRIALS=3
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2
```

#### Classic v0.2.4 and after

```bash
export RUN_ENTRIES_CONF_PATH=run_specs_lite.conf
export SCHEMA_PATH=schema_classic.yaml
export NUM_TRAIN_TRIALS=1
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2
```

#### Lite

```bash
export RUN_ENTRIES_CONF_PATH=run_entries_dec2023.conf
export SCHEMA_PATH=schema_lite.yaml
export NUM_TRAIN_TRIALS=1
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2
```

#### HEIM

```bash
export RUN_ENTRIES_CONF_PATH=run_specs_heim.conf
export SCHEMA_PATH=schema_heim.yaml
export NUM_TRAIN_TRIALS=1
export NUM_EVAL_INSTANCES=1000
export PRIORITY=2
```