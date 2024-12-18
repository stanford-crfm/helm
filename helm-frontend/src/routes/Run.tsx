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

  const [agreeInput, setAgreeInput] = useState("");
  const [userAgreed, setUserAgreed] = useState(false);

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

  // Handler for agreement
  const handleAgreement = () => {
    if (agreeInput.trim() === "Yes, I agree") {
      setUserAgreed(true);
    } else {
      setUserAgreed(false);
      alert("Please type 'Yes, I agree' exactly.");
    }
  };

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

      {activeTab === 0 && runName.includes("gpqa") && !userAgreed && (
        <div className="mb-8">
          <hr className="my-4" />
          <p className="mb-4">
            The GPQA dataset instances are encrypted by default to comply with
            the following request:
          </p>
          <blockquote className="italic border-l-4 border-gray-300 pl-4 text-gray-700 mb-4">
            “We ask that you do not reveal examples from this dataset in plain
            text or images online, to minimize the risk of these instances being
            included in foundation model training corpora.”
          </blockquote>
          <p className="mb-4">
            If you agree to this condition, please type{" "}
            <strong>"Yes, I agree"</strong> in the box below and then click{" "}
            <strong>Decrypt</strong>.
          </p>
          <div className="flex gap-2 mt-2">
            <input
              type="text"
              value={agreeInput}
              onChange={(e) => setAgreeInput(e.target.value)}
              className="input input-bordered"
              placeholder='Type "Yes, I agree"'
            />
            <button onClick={handleAgreement} className="btn btn-primary">
              Decrypt
            </button>
          </div>
          <hr className="my-4" />
        </div>
      )}

      {activeTab === 0 ? (
        <Instances
          key={userAgreed ? "instances-agreed" : "instances-not-agreed"}
          runName={runName}
          suite={runSuite}
          metricFieldMap={metricFieldMap}
          userAgreed={userAgreed} // Pass the boolean to Instances
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
