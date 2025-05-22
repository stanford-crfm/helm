import CompletionAnnotation from "./CompletionAnnotation";
import Perturbation from "./Perturbation";

export default interface DisplayPrediction {
  instance_id: string;
  predicted_text: string;
  train_trial_index: number;
  stats: Record<string, number>;
  mapped_output: string | undefined;
  base64_images?: string[] | undefined;
  thinking_text?: string | undefined;
  // beware you will have to update this for future custom annotations
  annotations?: Record<string, Array<CompletionAnnotation>> | undefined;
  perturbation?: Perturbation | undefined;
}
