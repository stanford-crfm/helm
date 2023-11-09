interface MetricSpec {
  class_name: string;
  args: {
    names: string[];
  };
}

export default interface RunSpec {
  name: string;
  scenario_spec: {
    class_name: string;
    args: {
      task?: string;
      subject?: string;
    };
  };
  adapter_spec: {
    method: string;
    global_prefix: string;
    instructions: string;
    input_prefix: string;
    input_suffix: string;
    reference_prefix: string;
    reference_suffix: string;
    output_prefix: string;
    output_suffix: string;
    instance_prefix: string;
    substitutions: string[];
    max_train_instances: number;
    max_eval_instances: number;
    num_outputs: number;
    num_train_trials: number;
    sample_train: boolean;
    model: string;
    temperature: number;
    max_tokens: number;
    stop_sequences: string[];
  };
  metric_specs: MetricSpec[];
  data_augmenter_spec: {
    perturbation_specs: unknown[];
    should_augment_train_instances: boolean;
    should_include_original_train: boolean;
    should_skip_unchanged_train: boolean;
    should_augment_eval_instances: boolean;
    should_include_original_eval: boolean;
    should_skip_unchanged_eval: boolean;
    seeds_per_instance: number;
  };
  groups: string[];
}
