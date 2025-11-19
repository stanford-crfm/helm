# HEIM (Text-to-image Model Evaluation)

**Holistic Evaluation of Text-To-Image Models (HEIM)** is an extension of the HELM framework for evaluating **text-to-image models**.

## Holistic Evaluation of Text-To-Image Models

<img src="https://github.com/stanford-crfm/helm/raw/heim/src/helm/benchmark/static/heim/images/heim-logo.png" alt=""  width="800"/>

Significant effort has recently been made in developing text-to-image generation models, which take textual prompts as 
input and generate images. As these models are widely used in real-world applications, there is an urgent need to 
comprehensively understand their capabilities and risks. However, existing evaluations primarily focus on image-text 
alignment and image quality. To address this limitation, we introduce a new benchmark, 
**Holistic Evaluation of Text-To-Image Models (HEIM)**.

We identify 12 different aspects that are important in real-world model deployment, including:

- image-text alignment
- image quality
- aesthetics
- originality
- reasoning
- knowledge
- bias
- toxicity
- fairness
- robustness
- multilinguality
- efficiency

By curating scenarios encompassing these aspects, we evaluate state-of-the-art text-to-image models using this benchmark. 
Unlike previous evaluations that focused on alignment and quality, HEIM significantly improves coverage by evaluating all 
models across all aspects. Our results reveal that no single model excels in all aspects, with different models 
demonstrating strengths in different aspects.

## References

- [Leaderboard](https://crfm.stanford.edu/helm/heim/latest/)
- [Paper](https://arxiv.org/abs/2311.04287)

## Installation

First, follow the [installation instructions](installation.md) to install the base HELM Python page.

To install the additional dependencies to run HEIM, run:

```sh
pip install "crfm-helm[heim]"
```

Some models (e.g., DALLE-mini/mega) and metrics (`DetectionMetric`) require extra dependencies that are 
not available on PyPI. To install these dependencies, download and run the 
[extra install script](https://github.com/stanford-crfm/helm/blob/main/install-heim-extras.sh):

```sh
bash install-heim-extras.sh
```

## Getting Started

The following is an example of evaluating [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) on the [MS-COCO scenario](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/scenarios/image_generation/mscoco_scenario.py) using 10 instances.

```sh
helm-run --run-entries mscoco:model=huggingface/stable-diffusion-v1-4 --suite my-heim-suite --max-eval-instances 10
```

## Reproducing the Leaderboard

To reproduce the [entire HEIM leaderboard](https://crfm.stanford.edu/helm/heim/latest/), refer to the instructions for HEIM on the [Reproducing Leaderboards](reproducing_leaderboards.md) documentation.

### Note:

The full HEIM leaderboard is not currently reproducible with these instructions. We are working to resolve this. In the meantime we have disabled the NSFWMetric to allow for the rest of the evaluation suite to run.
