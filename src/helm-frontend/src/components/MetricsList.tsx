import type Metric from "@/types/Metric";
import type MetricGroup from "@/types/MetricGroup";

interface Props {
  metrics: Metric[];
  metricGroups: MetricGroup[];
}

export default function MetricList({ metrics, metricGroups }: Props) {
  return (
    <section>
      <h3 className="text-3xl">{metrics.length} Metrics</h3>
      <ul>
        {metricGroups.map((metricGroup, idx) => (
          <li key={idx}>
            {metrics.filter((metric) =>
              metricGroup.metrics.some((m) => m.name === metric.name),
            ).length > 0 ? (
              <h4>
                {metricGroup.display_name} ({metricGroup.name})
              </h4>
            ) : null}
            <ul>
              {metrics
                .filter((metric) =>
                  metricGroup.metrics.some((m) => m.name === metric.name),
                )
                .map((metric, idx) => {
                  return (
                    <li key={idx} className="ml-4">
                      {metric.display_name}
                    </li>
                  );
                })}
            </ul>
          </li>
        ))}
      </ul>
    </section>
  );
}
