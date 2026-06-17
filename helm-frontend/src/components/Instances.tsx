import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import type Instance from "@/types/Instance";
import type MetricFieldMap from "@/types/MetricFieldMap";
import Loading from "@/components/Loading";
import InstanceData from "@/components/InstanceData";
import Pagination from "@/components/Pagination";
import DisplayPrediction from "@/types/DisplayPrediction";
import DisplayRequest from "@/types/DisplayRequest";
import getInstances from "@/services/getInstances";
import getDisplayPredictionsByName from "@/services/getDisplayPredictionsByName";
import getDisplayRequestsByName from "@/services/getDisplayRequestsByName";
import { matchesInstanceSearch } from "./instanceFilters";

const INSTANCES_PAGE_SIZE = 10;

declare global {
  interface Window {
    helmHasScrolledToInstance: boolean | undefined;
  }
}

interface Props {
  runName: string;
  suite: string;
  metricFieldMap: MetricFieldMap;
  userAgreed: boolean;
}

export default function Instances({
  runName,
  suite,
  metricFieldMap,
  userAgreed,
}: Props) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [instances, setInstances] = useState<Instance[]>([]);
  const [displayPredictionsMap, setDisplayPredictionsMap] = useState<
    undefined | Record<string, Record<string, DisplayPrediction[]>>
  >();
  const [displayRequestsMap, setDisplayRequestsMap] = useState<
    undefined | Record<string, Record<string, DisplayRequest[]>>
  >();
  const [currentInstancesPage, setCurrentInstancesPage] = useState<number>(1);
  const [instanceQuery, setInstanceQuery] = useState<string>(
    searchParams.get("instanceQuery") ?? "",
  );

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const signal = controller.signal;

      const [instancesResp, displayPredictions, displayRequests] =
        await Promise.all([
          getInstances(runName, signal, suite, userAgreed),
          getDisplayPredictionsByName(runName, signal, suite, userAgreed),
          getDisplayRequestsByName(runName, signal, suite, userAgreed),
        ]);
      setInstances(instancesResp);

      const tempDisplayRequestsMap: {
        [key: string]: { [key: string]: DisplayRequest[] };
      } = {};
      displayRequests.forEach((displayRequest) => {
        const instanceId = displayRequest.instance_id;
        const perturbationName = displayRequest.perturbation?.name || "";
        if (tempDisplayRequestsMap[instanceId] === undefined) {
          tempDisplayRequestsMap[instanceId] = {};
        }
        if (
          tempDisplayRequestsMap[instanceId][perturbationName] === undefined
        ) {
          tempDisplayRequestsMap[instanceId][perturbationName] = [];
        }
        tempDisplayRequestsMap[instanceId][perturbationName].push(
          displayRequest,
        );
      });
      setDisplayRequestsMap(tempDisplayRequestsMap);

      const tempDisplayPredictionsMap: {
        [key: string]: { [key: string]: DisplayPrediction[] };
      } = {};
      displayPredictions.forEach((displayPrediction) => {
        const instanceId = displayPrediction.instance_id;
        const perturbationName = displayPrediction.perturbation?.name || "";
        if (tempDisplayPredictionsMap[instanceId] === undefined) {
          tempDisplayPredictionsMap[instanceId] = {};
        }
        if (
          tempDisplayPredictionsMap[instanceId][perturbationName] === undefined
        ) {
          tempDisplayPredictionsMap[instanceId][perturbationName] = [];
        }
        tempDisplayPredictionsMap[instanceId][perturbationName].push(
          displayPrediction,
        );
      });
      setDisplayPredictionsMap(tempDisplayPredictionsMap);
    }

    void fetchData();

    return () => controller.abort();
  }, [runName, suite, userAgreed]);

  const filteredInstances = useMemo(
    () =>
      instances.filter((instance) =>
        matchesInstanceSearch(instance, instanceQuery),
      ),
    [instanceQuery, instances],
  );

  useEffect(() => {
    setCurrentInstancesPage(1);
  }, [instanceQuery]);

  const pagedInstances = filteredInstances.slice(
    (currentInstancesPage - 1) * INSTANCES_PAGE_SIZE,
    (currentInstancesPage - 1) * INSTANCES_PAGE_SIZE + INSTANCES_PAGE_SIZE,
  );
  const totalInstancesPages = Math.max(
    1,
    Math.ceil(filteredInstances.length / INSTANCES_PAGE_SIZE),
  );

  // Handle scrolling to anchored instance
  useEffect(() => {
    const anchoredInstance = searchParams.get("instance");
    if (
      anchoredInstance &&
      !window.helmHasScrolledToInstance &&
      pagedInstances.length > 0
    ) {
      // Find the index of the anchored instance
      const instanceIndex = pagedInstances.findIndex(
        (i) => i.id === anchoredInstance,
      );
      if (instanceIndex === -1) return;

      // Wait for the DOM to be updated with the correct page
      requestAnimationFrame(() => {
        const element = document.getElementById(`instance-${anchoredInstance}`);
        if (element) {
          element.scrollIntoView({ behavior: "smooth" });
        }
      });

      window.helmHasScrolledToInstance = true;
    }
  }, [searchParams, currentInstancesPage, setSearchParams, pagedInstances]);

  const renderInstanceId = (instance: Instance): string => {
    return instance.perturbation === undefined
      ? `Instance id: ${instance.id} [split: ${instance.split}]`
      : `Instance id: ${instance.id} [split: ${instance.split}][perturbation: ${instance.perturbation.name}]`;
  };

  if (displayPredictionsMap === undefined || displayRequestsMap === undefined) {
    return <Loading />;
  }

  return (
    <>
      <div className="mb-6">
        <label className="form-control w-full max-w-xl">
          <div className="label">
            <span className="label-text font-medium">Search instances</span>
          </div>
          <input
            className="input input-bordered w-full"
            type="search"
            placeholder="Filter by instance id, input text, or reference"
            value={instanceQuery}
            onChange={(event) => {
              const nextQuery = event.target.value;
              setInstanceQuery(nextQuery);

              if (nextQuery) {
                searchParams.set("instanceQuery", nextQuery);
              } else {
                searchParams.delete("instanceQuery");
              }

              searchParams.set("instancesPage", "1");
              setSearchParams(searchParams);
            }}
          />
        </label>
        <p className="mt-2 text-sm text-gray-600">
          Showing {filteredInstances.length} of {instances.length} instances.
        </p>
      </div>
      <div className="grid gap-8">
        {pagedInstances.map((instance, idx) => (
          <div id={"instance-" + instance.id} className="border p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl mb-4">{renderInstanceId(instance)}</h3>
              <button
                className="btn btn-sm normal-case px-2 py-1"
                onClick={() => {
                  const url =
                    window.location.href +
                    (window.location.href.includes("?")
                      ? "&instance="
                      : "?instance=") +
                    instance.id;
                  void navigator.clipboard.writeText(url);
                }}
              >
                Copy Link
              </button>
            </div>
            <InstanceData
              key={`${instance.id}-${idx}`}
              instance={instance}
              requests={
                displayRequestsMap[instance.id][
                  instance.perturbation?.name || ""
                ]
              }
              predictions={
                displayPredictionsMap[instance.id][
                  instance.perturbation?.name || ""
                ]
              }
              metricFieldMap={metricFieldMap}
            />
          </div>
        ))}
      </div>
      {filteredInstances.length === 0 ? (
        <p className="rounded-md border border-dashed p-4 text-sm text-gray-600">
          No instances match the current search.
        </p>
      ) : null}
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
  );
}
