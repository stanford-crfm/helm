# HEIM Quick Start (text-to-image evaluation)

To run HEIM, follow these steps:

1. Create a run specs configuration file or use [an existing one](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation) 
at `src/helm/benchmark/presentation`. For example, to evaluate 
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) against the 
[MS-COCO scenario](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/scenarios/image_generation/mscoco_scenario.py), run:
```bash
echo 'entries: [{description: "mscoco:model=huggingface/stable-diffusion-v1-4", priority: 1}]' > run_specs.conf
```
2. Run the benchmark on the run specs by specifying the path to the run specs configuration file from the previous step for 
`--conf-paths`. You can also specify certain number of instances (e.g., 10 instances): 
```bash
helm-run --conf-paths run_specs.conf --suite heim_v1 --max-eval-instances 10
```

Examples of run specs configuration files can be found [here](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation).
We used [this configuration file](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs_heim.conf) 
to produce results of the paper.


## Evaluate a local Diffusers checkpoints

To evaluate a local [Diffusers](https://huggingface.co/docs/diffusers/index) checkpoint, 
specify the path to the checkpoint when running `helm-run` with `--enable-local-diffusers-models <path to local checkpoint>`.
For example, if you wanted to evaluate a local checkpoint with the HEIM Lite conf file, you would run:

```bash
helm-run --suite heim_v1 --conf-paths src/helm/benchmark/presentation/run_specs_heim_lite.conf --max-eval-instances 100  --enable-local-diffusers-models /path/to/local/my_checkpoint --models-to-run huggingface/my_checkpoint
```

The model name in HELM would be `huggingface/` plus the name of the folder containing the checkpoint, 
so in the example above, the HELM model name would be `huggingface/my_checkpoint`.
