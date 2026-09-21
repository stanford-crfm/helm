import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import getStatsByName from "@/services/getStatsByName";
import type Stat from "@/types/Stat";
import type MetricFieldMap from "@/types/MetricFieldMap";
import Loading from "@/components/Loading";
import Pagination from "@/components/Pagination";
import StatNameDisplay from "@/components/StatNameDisplay";

const METRICS_PAGE_SIZE = 50;
const TABLE_COLUMN_NAMES = [
  "name",
  "mean",
  "min",
  "max",
  "sum",
  "sum_squared",
  "variance",
  "stddev",
];

interface Props {
  runName: string;
  suite: string;
  metricFieldMap: MetricFieldMap;
}

export default function Run({ runName, suite, metricFieldMap }: Props) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [stats, setStats] = useState<Stat[] | undefined>();
  const [currentMetricsPage, setCurrentMetricsPage] = useState<number>(1);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const signal = controller.signal;
      const statsResp = await getStatsByName(runName, signal, suite);
      setStats(statsResp);
    }

    void fetchData();

    return () => controller.abort();
  }, [runName, suite]);

  if (stats === undefined || stats.length === 0) {
    return <Loading />;
  }

  const totalMetricsPages = Math.ceil(stats.length / METRICS_PAGE_SIZE);

  const pagedMetrics = stats.slice(
    (currentMetricsPage - 1) * METRICS_PAGE_SIZE,
    (currentMetricsPage - 1) * METRICS_PAGE_SIZE + METRICS_PAGE_SIZE,
  );

  return (
    <div>
      {/* Search bar */}
      <div className="flex justify-start my-4">
        <input
          type="text"
          className="input input-bordered w-full max-w-xs"
          placeholder="Search for a metric"
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              {TABLE_COLUMN_NAMES.map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pagedMetrics
              .filter(
                (stat) =>
                  !searchTerm ||
                  stat.name.name
                    .toLowerCase()
                    .includes(searchTerm.toLowerCase()),
              )
              .map((stat) => (
                <tr>
                  {TABLE_COLUMN_NAMES.map((key) => {
                    const value = stat[key as keyof Stat];
                    if (typeof value === "number") {
                      return <td>{value}</td>;
                    } else {
                      return (
                        <td key={key}>
                          <StatNameDisplay
                            stat={stat}
                            metricFieldMap={metricFieldMap}
                          />
                        </td>
                      );
                    }
                  })}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <Pagination
        className="flex justify-center my-8"
        onNextPage={() => {
          const nextMetricsPage = Math.min(
            currentMetricsPage + 1,
            totalMetricsPages,
          );
          setCurrentMetricsPage(nextMetricsPage);
          searchParams.set("metricsPage", String(nextMetricsPage));
          setSearchParams(searchParams);
        }}
        onPrevPage={() => {
          const prevMetricsPage = Math.max(currentMetricsPage - 1, 1);
          setCurrentMetricsPage(prevMetricsPage);
          searchParams.set("metricsPage", String(prevMetricsPage));
          setSearchParams(searchParams);
        }}
        currentPage={currentMetricsPage}
        totalPages={totalMetricsPages}
      />
    </div>
  );
}
