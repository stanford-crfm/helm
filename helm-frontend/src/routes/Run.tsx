import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Badge, Card, List, ListItem } from "@tremor/react";
import {
  ArrowDownTrayIcon,
  ArrowTopRightOnSquareIcon,
} from "@heroicons/react/24/solid";
import getSchema from "@/services/getSchema";
import type RunSpec from "@/types/RunSpec";
import getScenarioByName from "@/services/getScenarioByName";
import type Scenario from "@/types/Scenario";
import type AdapterFieldMap from "@/types/AdapterFieldMap";
import type MetricFieldMap from "@/types/MetricFieldMap";
import getRunSpecByName, {
  getRunSpecByNameUrl,
} from "@/services/getRunSpecByName";
import { getScenarioStateByNameUrl } from "@/services/getScenarioStateByName";
import Tab from "@/components/Tab";
import Tabs from "@/components/Tabs";
import Loading from "@/components/Loading";
import Model from "@/types/Model";
import MarkdownValue from "@/components/MarkdownValue";
import getRunsToRunSuites from "@/services/getRunsToRunSuites";
import getSuiteForRun from "@/services/getSuiteForRun";
import Instances from "@/components/Instances";
import RunMetrics from "@/components/RunMetrics";

export default function Run() {
  const { runName } = useParams();
  const [activeTab, setActiveTab] = useState<number>(0);
  const [runSpec, setRunSpec] = useState<RunSpec | undefined>();
  const [runSuite, setRunSuite] = useState<string | undefined>();
  const [model, setModel] = useState<Model | undefined>();
  const [scenario, setScenario] = useState<Scenario | undefined>();
  const [adapterFieldMap, setAdapterFieldMap] = useState<AdapterFieldMap>({});
  const [metricFieldMap, setMetricFieldMap] = useState<
    MetricFieldMap | undefined
  >({});

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

      const [runSpecResp, scenario, schema] = await Promise.all([
        getRunSpecByName(runName, signal, suite),
        getScenarioByName(runName, signal, suite),
        getSchema(signal),
      ]);

      setRunSpec(runSpecResp);
      setScenario(scenario);

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
        schema.models.find((m) => m.name === runSpecResp?.adapter_spec.model),
      );
    }

    void fetchData();

    return () => controller.abort();
  }, [runName]);

  if (
    runSpec === undefined ||
    scenario === undefined ||
    runName === undefined ||
    runSuite === undefined ||
    metricFieldMap === undefined
  ) {
    return <Loading />;
  }

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
        <Instances
          runName={runName}
          suite={runSuite}
          metricFieldMap={metricFieldMap}
        />
      ) : (
        <RunMetrics
          runName={runName}
          suite={runSuite}
          metricFieldMap={metricFieldMap}
        />
      )}
    </>
  );
}
