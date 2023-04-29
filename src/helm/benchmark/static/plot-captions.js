////////////////////////////////////////////////////////////
// Dictionary of plot captions

const plotCaptions = {
    "generic_summary": "Metrics for every model on every core scenario as a means for indicating the spread on a per-metric basis.",
    "model_ranking_all": "The fraction of head-to-head comparisons between the given model and all other models, across all scenarios, where the given model is higher along the metric (e.g. more accurate in the accuracy subfigure). If a model was the highest for the given metric for every scenario, it would receive a score of 1.0; if a model received a score of 0.5, then if a scenario and second model were chosen at random, the outcome of the comparison would be a coin flip. For calibration error, we measure ECE-10; for bias, we measure bias in gender representation.",
    "accuracy_v_x": "The relationship between accuracy (x-axis) and each of the 6 metrics (calibration, robustness, fairness, social bias, toxicity, efficiency) we study in this work across all core scenarios and for all models. For calibration error, we measure ECE-10; for bias, we measure bias in gender representation; and for efficiency, we measure denoised inference time.",
    "metric_correlation": "The Pearson correlation between each metric and every other metric (x-axis). The small blue dots denote the correlation on each individual scenario, while the larger orange dots average the correlation across scenarios. Trends are qualitatively similarly for other correlation measures (e.g. Spearman correlation). For calibration error, we measure ECE-10; for bias, we measure bias in gender representation; and for efficiency, we measure denoised inference time.",
    "accuracy_v_access": "The relationship between access (open vs. limited vs.  closed) and model accuracy for each of the core scenarios. Shaded bars indicate the performance of the best model for that scenario, whereas the solid bars indicate the performance of the overall most accurate model across all core scenarios.",
    "accuracy_over_num_parameters": "Cumulative plot, depicting the accuracy of the most accurate model up to a given size across all core scenarios.",
    "accuracy_over_release_date": "The relationship between time (x-axis) and the accuracy of models (y-axis) across the core scenarios.",
    "accuracy_over_the_pile_perplexity": "The relationship between log bits-per-byte (BPB) on The Pile and the accuracy on each core scenario.",
    "targeted_evals": "Model accuracy on scenario targeting specific performance components (language, knowledge, reasoning).",
    "in_context_ablations": "For each model, we set the maximum number of in-context examples to [0, 1, 2, 4, 8, 16] and fit as many in-context examples as possible within the context window. We plot performance as a function of the average number of in-context examples actually used.",
    "mc_ablations": "For each adaptation method (joint, separate, and separate calibrated), we compare models across scenarios."
};
