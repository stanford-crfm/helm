import { useEffect, useState } from "react";
import { BarList, Card, Metric, Text } from "@tremor/react";
import type RunGroup from "@/types/RunGroup";
import getSchema from "@/services/getSchema";
import MarkdownValue from "@/components/MarkdownValue";
import PageTitle from "@/components/PageTitle";
import Link from "@/components/Link";
import Loading from "@/components/Loading";
import MetricField from "@/types/MetricField";

export default function Scenarios() {
  const [runGroups, setRunGroups] = useState<RunGroup[]>([] as RunGroup[]);
  const [nameToMetric, setNameToMetric] = useState<Record<string, MetricField>>(
    {},
  );

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schema = await getSchema(controller.signal);
      /**
       * Currently just filtering out on a couple
       * falsey values. This can likely be more robust
       */
      setRunGroups(
        schema.run_groups.filter(
          (runGroup) =>
            !runGroup.todo &&
            runGroup.taxonomy &&
            !runGroup.display_name.includes("CLEVA"),
        ),
      );

      setNameToMetric(
        Object.fromEntries(
          schema.metrics.map((metric) => [metric.name, metric]),
        ) as Record<string, MetricField>,
      );
    }
    void fetchData();

    return () => controller.abort();
  }, []);

  const getMetricForRunGroup = (runGroup: RunGroup): string => {
    const mainName: string | undefined = runGroup.environment?.main_name;
    const metric: MetricField | undefined = mainName
      ? nameToMetric[mainName]
      : undefined;
    if (metric === undefined) {
      return "";
    }
    const metricName =
      metric.display_name || metric.short_display_name || metric.name;
    return metric.description
      ? `${metricName} \u2013 ${metric.description}`
      : metricName;
  };

  const taskBuckets = Object.values(
    runGroups.reduce(
      (acc, runGroup) => {
        const task = runGroup.taxonomy?.task || "Unknown";
        if (acc[task] === undefined) {
          acc[task] = {
            name: task,
            value: 1,
          };

          return acc;
        }

        acc[task].value += 1;
        return acc;
      },
      {} as Record<string, { name: string; value: number }>,
    ),
  );

  if (runGroups.length === 0) {
    return <Loading />;
  }

  console.log(runGroups);

  return (
    <>
      <PageTitle
        title="Scenarios"
        subtitle="A scenario represents a use case and consists of a dataset of instances."
      />
      <div className="overflow-x-auto mt-12">
        <table className="table">
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Task</th>
              <th>What</th>
              <th>Who</th>
              <th>When</th>
              <th>Language</th>
              <th>Description</th>
              <th>Metric</th>
            </tr>
          </thead>
          <tbody>
            {runGroups.map((runGroup) => (
              <tr>
                <td>
                  <Link to={`/leaderboard/${runGroup.name}`}>
                    <span className="text-lg">{runGroup.display_name}</span>
                  </Link>
                  <span className="block">{runGroup.name}</span>
                </td>
                <td>{runGroup.taxonomy?.task || ""}</td>
                <td>{runGroup.taxonomy?.what || ""}</td>
                <td>{runGroup.taxonomy?.who || ""}</td>
                <td>{runGroup.taxonomy?.when || ""}</td>
                <td>{runGroup.taxonomy?.language || ""}</td>
                <td>
                  <MarkdownValue
                    value={runGroup.short_description || runGroup.description}
                  />
                </td>
                <td>
                  <MarkdownValue value={getMetricForRunGroup(runGroup)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <PageTitle title="Analysis" />
        <div className="grid md:grid-cols-4 gap-8">
          <Card className="flex flex-col">
            <Text>Total scenarios</Text>
            <Metric className="mx-auto my-6 md:mt-16 !text-[72px] md:!text-[96px]">
              {runGroups.length}
            </Metric>
          </Card>
          <Card className="col-span-3">
            <div className="grid md:grid-cols-2 gap-x-12">
              <BarList
                data={taskBuckets.slice(0, Math.floor(taskBuckets.length / 2))}
              />
              <BarList
                data={taskBuckets.slice(Math.ceil(taskBuckets.length / 2))}
              />
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
