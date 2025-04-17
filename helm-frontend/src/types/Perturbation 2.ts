export default interface Perturbation {
  [key: string]: string | number | boolean | undefined;
  name?: string;
  display_name?: string;
  description?: string;
  computed_on?: string;
  fairness?: boolean;
  name_file_path?: string;
  person_name_type?: string;
  preserve_gender?: boolean;
  prob?: number;
  robustness?: boolean;
  source_class?: string;
  target_class?: string;
}
