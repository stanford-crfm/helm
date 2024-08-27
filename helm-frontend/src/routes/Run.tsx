import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { Badge, Card, List, ListItem } from "@tremor/react";
import {
  ArrowDownTrayIcon,
  ArrowTopRightOnSquareIcon,
} from "@heroicons/react/24/solid";
import getSchema from "@/services/getSchema";
import getRunSpecs from "@/services/getRunSpecs";
import type RunSpec from "@/types/RunSpec";
import getInstances from "@/services/getInstances";
import type Instance from "@/types/Instance";
import getStatsByName from "@/services/getStatsByName";
import type Stat from "@/types/Stat";
import getDisplayRequestsByName from "@/services/getDisplayRequestsByName";
import type DisplayRequestsMap from "@/types/DisplayRequestsMap";
import getDisplayPredictionsByName from "@/services/getDisplayPredictionsByName";
import type DisplayPredictionsMap from "@/types/DisplayPredictionsMap";
import getScenarioByName from "@/services/getScenarioByName";
import type Scenario from "@/types/Scenario";
import type AdapterFieldMap from "@/types/AdapterFieldMap";
import type MetricFieldMap from "@/types/MetricFieldMap";
import { getRunSpecByNameUrl } from "@/services/getRunSpecByName";
import { getScenarioStateByNameUrl } from "@/services/getScenarioStateByName";
import Tab from "@/components/Tab";
import Tabs from "@/components/Tabs";
import InstanceData from "@/components/InstanceData";
import Loading from "@/components/Loading";
import Pagination from "@/components/Pagination";
import Model from "@/types/Model";
import MarkdownValue from "@/components/MarkdownValue";
import StatNameDisplay from "@/components/StatNameDisplay";
import getRunsToRunSuites from "@/services/getRunsToRunSuites";
import getSuiteForRun from "@/services/getSuiteForRun";

const INSTANCES_PAGE_SIZE = 10;
const METRICS_PAGE_SIZE = 50;

export default function Run() {
  const { runName } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState<number>(0);
  const [runSpec, setRunSpec] = useState<RunSpec | undefined>();
  const [runSuite, setRunSuite] = useState<string | undefined>();
  const [instances, setInstances] = useState<Instance[]>([]);
  const [stats, setStats] = useState<Stat[]>([]);
  const [displayPredictionsMap, setDisplayPredictionsMap] = useState<
    DisplayPredictionsMap | undefined
  >();
  const [displayRequestsMap, setDisplayRequestsMap] = useState<
    DisplayRequestsMap | undefined
  >();
  const [currentInstancesPage, setCurrentInstancesPage] = useState<number>(1);
  const [totalInstancesPages, setTotalInstancesPages] = useState<number>(1);
  const [currentMetricsPage, setCurrentMetricsPage] = useState<number>(1);
  const [totalMetricsPages, setTotalMetricsPages] = useState<number>(1);
  const [model, setModel] = useState<Model | undefined>();
  const [scenario, setScenario] = useState<Scenario | undefined>();
  const [adapterFieldMap, setAdapterFieldMap] = useState<AdapterFieldMap>({});
  const [metricFieldMap, setMetricFieldMap] = useState<MetricFieldMap>({});
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const signal = controller.signal;

      if (runName === undefined) {
        return () => controller.abort();
      }

      const suite = window.SUITE
        ? window.SUITE
        : getSuiteForRun(await getRunsToRunSuites(signal), runName);
      setRunSuite(suite);

      const [
        runSpecs,
        instancesResp,
        statsResp,
        scenario,
        displayPredictions,
        displayRequests,
      ] = await Promise.all([
        getRunSpecs(signal),
        getInstances(runName, signal, suite),
        getStatsByName(runName, signal, suite),
        getScenarioByName(runName, signal, suite),
        getDisplayPredictionsByName(runName, signal, suite),
        getDisplayRequestsByName(runName, signal, suite),
      ]);

      setRunSpec(runSpecs.find((rs) => rs.name === runName));
      setInstances(instancesResp);
      const totalInstancesPages = Math.ceil(
        instancesResp.length / INSTANCES_PAGE_SIZE,
      );
      const instancePage = Number(searchParams.get("instancesPage") || 1);
      setTotalInstancesPages(totalInstancesPages);
      setCurrentInstancesPage(
        Math.max(Math.min(instancePage, totalInstancesPages), 1),
      );
      setStats(statsResp);
      setScenario(scenario);
      const totalMetricsPages = Math.floor(
        statsResp.length / METRICS_PAGE_SIZE,
      );
      const metricPage = Number(searchParams.get("metricsPage") || 1);
      setTotalMetricsPages(totalMetricsPages);
      setCurrentMetricsPage(
        Math.max(Math.min(metricPage, totalMetricsPages), 1),
      );
      setDisplayPredictionsMap(
        displayPredictions.reduce((acc, cur) => {
          if (acc[cur.instance_id] === undefined) {
            acc[cur.instance_id] = [];
          }
          acc[cur.instance_id].push(cur);
          return acc;
        }, {} as DisplayPredictionsMap),
      );
      setDisplayRequestsMap(
        displayRequests.reduce((acc, cur) => {
          if (acc[cur.instance_id] === undefined) {
            acc[cur.instance_id] = [];
          }
          acc[cur.instance_id].push(cur);
          return acc;
        }, {} as DisplayRequestsMap),
      );
      const schema = await getSchema(signal);

      setMetricFieldMap(
        schema.metrics.reduce((acc, cur) => {
          acc[cur.name] = cur;
          return acc;
        }, {} as MetricFieldMap),
      );
      setAdapterFieldMap(
        schema.adapter.reduce((acc, cur) => {
          acc[cur.name] = cur;
          return acc;
        }, {} as AdapterFieldMap),
      );

      setModel(
        schema.models.find(
          (m) =>
            m.name ===
            runSpecs.find((rs) => rs.name === runName)?.adapter_spec.model,
        ),
      );
    }

    void fetchData();

    return () => controller.abort();
  }, [runName, searchParams]);

  if (
    runSpec === undefined ||
    displayPredictionsMap === undefined ||
    displayRequestsMap === undefined ||
    scenario === undefined
  ) {
    return <Loading />;
  }

  const pagedInstances = instances.slice(
    (currentInstancesPage - 1) * INSTANCES_PAGE_SIZE,
    (currentInstancesPage - 1) * INSTANCES_PAGE_SIZE + INSTANCES_PAGE_SIZE,
  );

  const pagedMetrics = stats.slice(
    (currentMetricsPage - 1) * METRICS_PAGE_SIZE,
    (currentMetricsPage - 1) * METRICS_PAGE_SIZE + METRICS_PAGE_SIZE,
  );

  return (
    <>
      <div className="flex justify-between gap-8 mb-12">
        <div>
          <h1 className="text-3xl flex items-center">
            {scenario.name}
            <a href={"/#/groups/" + scenario.name}>
              <ArrowTopRightOnSquareIcon className="w-6 h-6 ml-2" />
            </a>
          </h1>
          <h3 className="text-xl">
            <MarkdownValue value={scenario.description} />
          </h3>
          <h1 className="text-3xl mt-2">{runSpec.adapter_spec.model}</h1>
          <h3 className="text-xl">
            <MarkdownValue value={model?.description || ""} />
          </h3>
          <div className="mt-2 flex gap-2">
            {scenario.tags.map((tag) => (
              <Badge size="xs" color="gray">
                <span className="text text-md">{tag}</span>
              </Badge>
            ))}
          </div>
        </div>
      </div>
      <Card>
        <div className="flex justify-between">
          <h3 className="text-lg mb-1">Adapter Specification</h3>
          <div className="flex gap-2">
            <ArrowDownTrayIcon className="w-6 h-6 mr-1 text text-primary" />
            <a
              className="link link-primary link-hover"
              href={getRunSpecByNameUrl(runSpec.name, runSuite)}
              download="true"
              target="_blank"
            >
              Spec JSON
            </a>
            <a
              className="link link-primary link-hover"
              href={getScenarioStateByNameUrl(runSpec.name, runSuite)}
              download="true"
              target="_blank"
            >
              Full JSON
            </a>
          </div>
        </div>
        <div>
          <List className="grid md:grid-cols-2 lg:grid-cols-3 gap-x-8">
            {Object.entries(runSpec.adapter_spec).map(([key, value], idx) => (
              <ListItem className={idx < 3 ? "!border-0" : ""}>
                <strong
                  className="mr-1"
                  title={
                    adapterFieldMap[key]
                      ? adapterFieldMap[key].description
                      : undefined
                  }
                >{`${key}: `}</strong>
                <span className="overflow-x-auto">{value}</span>
              </ListItem>
            ))}
          </List>
        </div>
      </Card>
      <div className="mt-16 mb-8">
        <Tabs>
          <Tab
            size="lg"
            active={activeTab === 0}
            onClick={() => setActiveTab(0)}
          >
            Instances + Predictions
          </Tab>
          <Tab
            size="lg"
            active={activeTab === 1}
            onClick={() => setActiveTab(1)}
          >
            All metrics
          </Tab>
        </Tabs>
      </div>
      {activeTab === 0 ? (
        <>
          <div className="grid gap-8">
            {pagedInstances.map((instance, idx) => (
              <InstanceData
                key={`${instance.id}-${idx}`}
                instance={instance}
                requests={displayRequestsMap[instance.id]}
                predictions={displayPredictionsMap[instance.id]}
                metricFieldMap={metricFieldMap}
              />
            ))}
          </div>
          <Pagination
            className="flex justify-center my-8"
            onNextPage={() => {
              const nextInstancePage = Math.min(
                currentInstancesPage + 1,
                totalInstancesPages,
              );
              setCurrentInstancesPage(nextInstancePage);
              searchParams.set("instancesPage", String(nextInstancePage));
              setSearchParams(searchParams);
            }}
            onPrevPage={() => {
              const prevInstancePage = Math.max(currentInstancesPage - 1, 1);
              setCurrentInstancesPage(prevInstancePage);
              searchParams.set("instancesPage", String(prevInstancePage));
              setSearchParams(searchParams);
            }}
            currentPage={currentInstancesPage}
            totalPages={totalInstancesPages}
          />
        </>
      ) : (
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
                  {Object.keys(stats[0]).map((key) => (
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
                      {Object.entries(stat).map(([key, value]) => {
                        if (key === "name") {
                          return (
                            <td key={key}>
                              <StatNameDisplay
                                stat={stat}
                                metricFieldMap={metricFieldMap}
                              />
                            </td>
                          );
                        }
                        return <td>{value}</td>;
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
      )}
    </>
  );
}
