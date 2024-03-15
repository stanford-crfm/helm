import type AdapterField from "@/types/AdapterField";
import type MetricGroup from "@/types/MetricGroup";
import type MetricField from "@/types/MetricField";
import type Model from "@/types/Model";
import type Perturbation from "@/types/Perturbation";
import type RunGroup from "@/types/RunGroup";

export default interface Schema {
  adapter: AdapterField[];
  metric_groups: MetricGroup[];
  metrics: MetricField[];
  models: Model[];
  perturbations: Perturbation[];
  run_groups: RunGroup[];
}
