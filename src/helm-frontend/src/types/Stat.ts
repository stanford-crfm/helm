export default interface Stat {
  count: number;
  max: number;
  mean: number;
  min: number;
  name: {
    name: string;
    split?: string;
    sub_split?: string;
    perturbation?: {
      computed_on: string;
      fairness: boolean;
      name: string;
      name_file_path: string;
      person_name_type: string;
      preserve_gender: boolean;
      prob: number;
      robustness: boolean;
      source_class: string;
      target_class: string;
    };
  };
  stddev: number;
  sum: number;
  sum_squared: number;
  variance: number;
}
