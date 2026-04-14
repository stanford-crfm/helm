# LICA-Bench (HELM integration)

[LICA-Bench](https://github.com/purvanshi/lica-bench) evaluates vision-language models on graphic design tasks (layout, typography, SVG, templates, temporal/video, Lottie, and category understanding). HELM exposes each task as the scenario **`lica_bench`** so you can run evaluations through `helm-run` alongside other benchmarks.

## Setup

1. Install HELM with the optional extra:

   ```bash
   pip install "crfm-helm[lica-bench]"
   ```

2. Download the dataset bundle (contains `lica-data/` and `benchmarks/`). From the lica-bench repository:

   ```bash
   python scripts/download_data.py
   ```

   Or unpack the release zip so you have a single root directory (referred to below as `lica-benchmarks-dataset`).

3. Point HELM at that root, either:

   - **Environment variable:** `export LICA_BENCH_DATASET_ROOT=/path/to/lica-benchmarks-dataset`
   - **Run entry:** pass `dataset_root=...` (avoid commas in the path; they separate run-entry fields).

## Run entries

Run spec name: **`lica_bench`**.

Arguments:

| Argument        | Required | Description |
|-----------------|----------|-------------|
| `benchmark_id`  | yes      | Task id (`category-1`, `svg-3`, …). List tasks with lica-bench’s `scripts/run_benchmarks.py --list`. |
| `dataset_root`  | no       | Overrides `LICA_BENCH_DATASET_ROOT` if set. |
| `max_instances` | no       | Limit instances (useful for smoke tests). |

Example (with env var set):

```text
lica_bench:benchmark_id=category-1,model=openai/gpt-4o
```

Example (explicit root):

```text
lica_bench:benchmark_id=svg-1,dataset_root=/data/lica-benchmarks-dataset,model=openai/gpt-4o
```

## Metrics

HELM reports generic **text generation** metrics (exact match, quasi-exact match, F1, ROUGE, BLEU, CIDEr). Many lica-bench tasks define **task-specific** scores (e.g. top-5 accuracy, layout IoU, generation quality). For those official numbers, use the native lica-bench CLI or Python API as documented in the [lica-bench README](https://github.com/purvanshi/lica-bench).

## Citation

If you use LICA-Bench or the Lica dataset, cite the dataset as indicated in the [lica-bench repository](https://github.com/purvanshi/lica-bench#citation).
