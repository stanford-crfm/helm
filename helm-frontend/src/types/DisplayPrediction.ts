import CompletionAnnotation from "./CompletionAnnotation";

export default interface DisplayPrediction {
  instance_id: string;
  predicted_text: string;
  train_trial_index: number;
  stats: {
    num_output_tokens: number;
    num_prompt_tokens: number;
    num_train_instances: number;
    num_train_trials: number;
    prompt_truncated: number;
    quasi_exact_match: number;
  };
  mapped_output: string | undefined;
  base64_images?: string[] | undefined;
  // beware you will have to update this for future custom annotations
  annotations?: Record<string, Array<CompletionAnnotation>> | undefined;
}
