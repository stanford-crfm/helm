# Quick Start

Run the following:

```
# Create a run specs configuration
echo 'entries: [{description: "mmlu:subject=philosophy,model=openai/gpt2", priority: 1}]' > run_entries.conf

# Run benchmark
helm-run --conf-paths run_entries.conf --suite v1 --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite v1

# Start a web server to display benchmark results
helm-server
```

Then go to http://localhost:8000/ in your browser.


## Next steps

Click [here](get_helm_rank.md) to find out how to run the full benchmark and get your model's leaderboard rank.

For the quick start page for HEIM, visit [here](heim.md).