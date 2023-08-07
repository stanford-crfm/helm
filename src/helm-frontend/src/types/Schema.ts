import type Adapter from "@/types/Adapter";
import type Metric from "@/types/Metric";
import type MetricGroup from "@/types/MetricGroup";
import type Model from "@/types/Model";
import type Peturbation from "@/types/Peturbation";
import type RunGroup from "@/types/RunGroup";

export default interface Schema {
  adapter: Adapter[];
  metric_groups: MetricGroup[];
  metrics: Metric[];
  models: Model[];
  perturbations: Peturbation[];
  run_groups: RunGroup[];
}
