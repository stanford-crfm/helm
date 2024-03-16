import type Stat from "@/types/Stat";
import type MetricFieldMap from "@/types/MetricFieldMap";

interface Props {
  stat: Stat;
  metricFieldMap: MetricFieldMap;
}

export default function StatNameDisplay({ stat, metricFieldMap }: Props) {
  const value = `${
    stat.name.split !== undefined ? ` on ${stat.name.split}` : ""
  }${stat.name.sub_split !== undefined ? `/${stat.name.sub_split}` : ""}${
    stat.name.perturbation !== undefined
      ? ` with ${stat.name.perturbation.name}`
      : " original"
  }`;
  return metricFieldMap[stat.name.name] ? (
    <span title={metricFieldMap[stat.name.name].description}>
      <strong>
        {metricFieldMap[stat.name.name].display_name || stat.name.name}
      </strong>
      {value}
    </span>
  ) : (
    <span>
      <strong>{stat.name.name}</strong>
      {value}
    </span>
  );
}
