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


## HEIM (text-to-image evaluation)

To run HEIM, follow these steps:

1. Create a run specs configuration file. For example, to evaluate 
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) against the 
[MS-COCO scenario](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/scenarios/image_generation/mscoco_scenario.py), run:
```
echo 'entries: [{description: "mscoco:model=huggingface/stable-diffusion-v1-4", priority: 1}]' > run_specs.conf
```
2. Run the benchmark with certain number of instances (e.g., 10 instances): 
`helm-run --conf-paths run_specs.conf --suite heim_v1 --max-eval-instances 10`

Examples of run specs configuration files can be found [here](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation).
We used [this configuration file](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs_heim.conf) 
to produce results of the paper.


## Next steps

Click [here](get_helm_rank.md) to find out how to to run the full benchmark and get your model's leaderboard rank.
