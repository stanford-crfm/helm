export default interface DisplayRequest {
  instance_id: string;
  train_trial_index: number;
  request: {
    echo_prompt: boolean;
    embedding: boolean;
    frequency_penalty: number;
    max_tokens: number;
    model: string;
    num_completions: number;
    presence_penalty: number;
    prompt: string;
    stop_sequences: string[];
    temperature: number;
    top_k_per_token: number;
    top_p: number;
  };
}
