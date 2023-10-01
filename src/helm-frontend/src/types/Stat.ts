import type Perturbation from "@/types/Perturbation";

export default interface Stat {
  count: number;
  max: number;
  mean: number;
  min: number;
  name: {
    name: string;
    split?: string;
    sub_split?: string;
    perturbation?: Perturbation;
  };
  stddev: number;
  sum: number;
  sum_squared: number;
  variance: number;
}
