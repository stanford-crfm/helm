import type Metric from "@/types/Metric";
import type MetricGroup from "@/types/MetricGroup";

interface Props {
  metrics: Metric[];
  metricGroups: MetricGroup[];
}

export default function MetricList({ metrics, metricGroups }: Props) {
  const metricNameToMetric = new Map<string, Metric>();
  metrics.forEach((metric) => metricNameToMetric.set(metric.name, metric));

  let numMetrics = 0;
  const metricGroupsWithMetrics: [MetricGroup, Metric[]][] = [];
  metricGroups.forEach((metricGroup) => {
    const metricGroupMetrics: Metric[] = [];
    metricGroup.metrics.forEach((metricField) => {
      const maybeMetric = metricNameToMetric.get(metricField.name);
      if (maybeMetric) {
        metricGroupMetrics.push(maybeMetric);
      }
    });
    if (metricGroupMetrics.length > 0) {
      metricGroupsWithMetrics.push([metricGroup, metricGroupMetrics]);
      numMetrics += metricGroupMetrics.length;
    }
  });

  return (
    <section>
      <h3 className="text-3xl">{numMetrics} metrics</h3>
      <ul>
        {metricGroupsWithMetrics.map(([metricGroup, metrics]) => (
          <li className="my-3" key={metricGroup.name}>
            <h4>{metricGroup.display_name}</h4>
            <ul className="list-disc list-inside">
              {metrics.map((metric) => {
                return (
                  <li key={metric.name} className="ml-4">
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
