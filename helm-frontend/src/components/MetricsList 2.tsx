import type MetricField from "@/types/MetricField";
import type MetricFieldMap from "@/types/MetricFieldMap";
import type MetricGroup from "@/types/MetricGroup";

interface Props {
  metricFieldMap: MetricFieldMap;
  metricGroups: MetricGroup[];
}

export default function MetricList({ metricFieldMap, metricGroups }: Props) {
  // Only count metrics that have a group and are displayed
  // i.e. don't count "orphaned" metrics
  // Also, don't double-count metrics that appear in multiple groups
  const groupedMetricNames = new Set<string>();

  const metricGroupsWithMetrics: [MetricGroup, MetricField[]][] = [];
  metricGroups.forEach((metricGroup) => {
    const metricGroupMetrics: MetricField[] = [];
    metricGroup.metrics.forEach((metricField) => {
      const maybeMetric = metricFieldMap[metricField.name];
      if (maybeMetric) {
        metricGroupMetrics.push(maybeMetric);
        groupedMetricNames.add(maybeMetric.name);
      }
    });
    if (metricGroupMetrics.length > 0) {
      metricGroupsWithMetrics.push([metricGroup, metricGroupMetrics]);
    }
  });

  return (
    <section>
      <h3 className="text-3xl">{groupedMetricNames.size} metrics</h3>
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
