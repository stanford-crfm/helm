# HEIM Quick Start (text-to-image evaluation)

To run HEIM, follow these steps:

1. Create a run specs configuration file. For example, to evaluate 
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) against the 
[MS-COCO scenario](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/scenarios/image_generation/mscoco_scenario.py), run:
```
echo 'entries: [{description: "mscoco:model=huggingface/stable-diffusion-v1-4", priority: 1}]' > run_entries.conf
```
2. Run the benchmark with certain number of instances (e.g., 10 instances): 
`helm-run --conf-paths run_entries.conf --suite heim_v1 --max-eval-instances 10`

Examples of run specs configuration files can be found [here](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation).
We used [this configuration file](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_entries_heim.conf) 
to produce results of the paper.
