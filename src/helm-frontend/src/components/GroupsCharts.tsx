import { useEffect, useState } from "react";
import type GroupsTable from "@/types/GroupsTable";
// @ts-ignore
import daisyuiColors from "daisyui/src/theming/themes";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Payload as LegendPayload } from "recharts/types/component/DefaultLegendContent";
import type { DataKey } from "recharts/types/util/types";

const colorNames = [
  "primary",
  "secondary",
  "accent",
  "success",
  "info",
  "warning",
  "error",
];
// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-assertion
const theme = daisyuiColors["[data-theme=business]"] as Record<string, string>;

interface Props {
  groupsTables: GroupsTable[];
  activeGroup: number;
}

function transformData(groupsTables: GroupsTable[], activeGroup: number) {
  return groupsTables[activeGroup].rows.map((row) => ({
    ...row.slice(1).reduce(
      (acc, cur, idx) => {
        acc[groupsTables[activeGroup].header[idx + 1].value] = isNaN(
          Number(cur.value),
        )
          ? 0
          : cur.value;

        return acc;
      },
      {} as {
        [key: string]: string | number;
      },
    ),
    ...{ name: row[0].value },
  }));
}

export default function GroupsCharts({ groupsTables, activeGroup }: Props) {
  const data = transformData(groupsTables, activeGroup);
  const [activeBars, setActiveBars] = useState<Map<string, boolean>>(() => {
    const map = new Map<string, boolean>();
    Object.keys(data[0]).forEach((key) => {
      map.set(key, true);
    });

    return map;
  });
  useEffect(() => {
    setActiveBars((prev) => {
      const map = new Map(prev);
      Object.keys(data[0]).forEach((key) => {
        if (map.has(key)) {
          map.set(key, Boolean(map.get(key)));
          return;
        }
        map.set(key, true);
      });

      return map;
    });
  }, [activeGroup, data]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleLegendClick = (e: LegendPayload & { dataKey?: DataKey<any> }) => {
    const { dataKey } = e;
    if (dataKey === undefined || typeof dataKey !== "string") {
      return;
    }
    setActiveBars((prev) => {
      const map = new Map(prev);
      map.set(dataKey, !prev.get(dataKey));

      return map;
    });
  };

  const sections = groupsTables[activeGroup].header
    .slice(1)
    .map((header) => header.value);

  /**
   * Adjusting height of the chart to fit aggll the
   * bars evenly. So that the individual bars are
   * about the same size regardless of the number
   * of total data points.
   */
  const height =
    10 *
    data.length *
    Math.min(
      [...activeBars.values()].filter((v) => v).length,
      Object.keys(data[0]).length - 1,
    );

  return (
    <div style={{ height: `${height}px` }} className="w-full">
      <ResponsiveContainer className="h-full w-full">
        <BarChart layout="vertical" width={500} height={height} data={data}>
          <Legend
            onClick={handleLegendClick}
            iconType="circle"
            verticalAlign="top"
            align="left"
          />
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={180} />
          <Tooltip />
          {sections.map((section, idx) => (
            <Bar
              hide={activeBars.get(section) === false}
              type="linear"
              fill={theme[colorNames[idx % colorNames.length]]}
              dataKey={section}
              key={section}
              name={section}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
