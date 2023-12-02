# Quick Start

Run the following:

```
# Create a run specs configuration
echo 'entries: [{description: "mmlu:subject=philosophy,model=huggingface/gpt2", priority: 1}]' > run_specs.conf

# Run benchmark
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite v1

# Start a web server to display benchmark results
helm-server
```

Then go to http://localhost:8000/ in your browser.

**Next steps:** click [here](get_helm_rank.md) to find out how to to run the full benchmark and get your model's leaderboard rank.