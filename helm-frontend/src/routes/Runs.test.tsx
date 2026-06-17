import { describe, expect, it } from "vitest";

import type RunSpec from "@/types/RunSpec";
import { filterRunSpecs } from "@/routes/Runs";

const makeRunSpec = (name: string, groups: string[]): RunSpec =>
  ({
    name,
    groups,
    scenario_spec: { class_name: "", args: {} },
    adapter_spec: {
      method: "",
      global_prefix: "",
      instructions: "",
      input_prefix: "",
      input_suffix: "",
      reference_prefix: "",
      reference_suffix: "",
      output_prefix: "",
      output_suffix: "",
      instance_prefix: "",
      substitutions: [],
      max_train_instances: 0,
      max_eval_instances: 0,
      num_outputs: 0,
      num_train_trials: 0,
      sample_train: false,
      model: "",
      temperature: 0,
      max_tokens: 0,
      stop_sequences: [],
    },
    metric_specs: [],
    data_augmenter_spec: {
      perturbation_specs: [],
      should_augment_train_instances: false,
      should_include_original_train: false,
      should_skip_unchanged_train: false,
      should_augment_eval_instances: false,
      should_include_original_eval: false,
      should_skip_unchanged_eval: false,
      seeds_per_instance: 0,
    },
  }) as RunSpec;

describe("filterRunSpecs", () => {
  it("filters by run name and group", () => {
    const runSpecs = [
      makeRunSpec("model-a,scenario-one", ["scenario_one"]),
      makeRunSpec("model-a,scenario-two", ["scenario_two"]),
      makeRunSpec("model-b,scenario-one", ["scenario_one"]),
    ];

    expect(filterRunSpecs(runSpecs, "model-a", false, "scenario_one")).toEqual([
      runSpecs[0],
    ]);
  });
});
