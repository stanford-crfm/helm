import type Metric from "@/types/Metric";
import type MetricGroup from "@/types/MetricGroup";
import { Link as ReactRouterLink } from "react-router-dom";

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
              <ReactRouterLink
                className="text-black"
                to={"groups/" + metricGroup.name}
              >
                <h4>{metricGroup.display_name}</h4>
              </ReactRouterLink>
            ) : null}
            <ul className="list-disc list-inside">
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
