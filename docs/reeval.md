# Reliable and Efficient Amortized Model-based Evaluation

Reliable and Efficient Amortized Model-based Evaluation (REEval) is an extension of the HELM framework for using Computerized Adaptive Testing (CAT) within the framework of Item Response Theory (IRT) to adaptively evaluate Large Language Models (LLMs). This approach selects the next question whose difficulty is closest to the estimated model ability, thereby reliably and efficiently eliciting the model's ability. The difficulties of the questions are provided on HuggingFace: [`stair-lab/reeval-difficulty-for-helm`](https://huggingface.co/datasets/stair-lab/reeval-difficulty-for-helm), which currently supports 22 scenarios in HELM. The paper's authors will supply a Python package for calculating these difficulties and will support more scenarios in the future.

# References

[Paper](https://arxiv.org/abs/2503.13335)

# Getting Started

The following is an example of adaptively evaluating Openai GPT2 on the MMLU scenario using 50 instances. The argument `--model-ability` is the initial ability of the model for reeval evaluation. The argument `--max-eval-instances` is the maximum number of samples to evaluate in the reeval mode. Other arguments stay the same as HELM.

```
# Run benchmark
export SUITE_NAME=reeval_mmlu_openai_gpt2
export MODELS_TO_RUN=openai/gpt2
export RUN_ENTRIES_CONF_PATH=run_entries_mmlu.conf
export SCHEMA_PATH=schema_mmlu.yaml
export NUM_TRAIN_TRIALS=1
export PRIORITY=4
export MODEL_ABILITY=0.0
export MAX_EVAL_INSTANCES=50
python3 -m helm.benchmark.reeval_run --conf-paths $RUN_ENTRIES_CONF_PATH --num-train-trials $NUM_TRAIN_TRIALS --priority $PRIORITY --suite $SUITE_NAME --models-to-run $MODELS_TO_RUN --model-ability $MODEL_ABILITY --max-eval-instances $MAX_EVAL_INSTANCES

# Summarize benchmark results
helm-summarize --schema $SCHEMA_PATH --suite $SUITE_NAME

# Start a web server to display benchmark results
helm-server --suite $SUITE_NAME
```

Then go to http://localhost:8000/ in your browser.
