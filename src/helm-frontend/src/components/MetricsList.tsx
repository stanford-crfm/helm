import type Metric from "@/types/Metric";
import type MetricGroup from "@/types/MetricGroup";

interface Props {
  metrics: Metric[];
  metricGroups: MetricGroup[];
}

export default function MetricList({ metrics, metricGroups }: Props) {
  return (
    <ul>
      <h3 className="text-3xl">{metrics.length} Metrics</h3>
      {metricGroups.map((metricGroup, idx) => (
        <li key={idx}>
          <ul>
            {metrics.filter((metric) =>
                metricGroup.metrics.some((m) => m.name === metric.name)
              ).length > 0
              ? (
                <h2>
                  {metricGroup.display_name} ({metricGroup.name})
                </h2>
              )
              : null}
            {metrics
              .filter((metric) =>
                metricGroup.metrics.some((m) => m.name === metric.name)
              )
              .map((metric, idx) => {
                return (
                  <li key={idx}>
                    {metric.display_name}
                  </li>
                );
              })}
          </ul>
        </li>
      ))}
    </ul>
  );
}
