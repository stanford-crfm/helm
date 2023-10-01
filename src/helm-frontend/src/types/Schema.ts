import type Adapter from "@/types/Adapter";
import type Metric from "@/types/Metric";
import type MetricGroup from "@/types/MetricGroup";
import type Model from "@/types/Model";
import type Perturbation from "@/types/Perturbation";
import type RunGroup from "@/types/RunGroup";

export default interface Schema {
  adapter: Adapter[];
  metric_groups: MetricGroup[];
  metrics: Metric[];
  models: Model[];
  perturbations: Perturbation[];
  run_groups: RunGroup[];
}
