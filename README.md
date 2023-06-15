<!--intro-start-->
# Holistic Evaluation of Text-To-Image Models

Significant progress has recently been made in developing text-to-image generation models. As these models are widely used in real-world applications, there is an urgent need to comprehensively understand their capabilities and risks. However, existing evaluations primarily focus on image-text alignment and quality. To address this limitation, we introduce a new benchmark, **Holistic Evaluation of Text-To-Image Models (HEIM).**

We identify 12 different aspects that are important in real-world model deployment, including image-text alignment, image quality, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality, and efficiency. By curating scenarios encompassing these aspects, we evaluate state-of-the-art text-to-image models using this benchmark. Unlike previous evaluations that focused on alignment and quality, HEIM significantly improves coverage by evaluating all models across all aspects. Our results reveal that no single model excels in all aspects, with different models demonstrating strengths in different aspects.

This repository branch contains the code used to produce the [main results of the paper](https://crfm.stanford.edu/heim/latest/).

In the future, we intend to merge this back into the main [Holistic Evaluation of Language Models (HELM) package](https://github.com/stanford-crfm/helm/), so that users can use this framework to evaluate their own text-to-image models, scenarios, and metrics.

<!--intro-end-->
