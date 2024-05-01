import { useEffect, useState } from "react";
import getReleaseSummary from "@/services/getReleaseSummary";
import ReleaseSummary from "@/types/ReleaseSummary";
import ProjectMetadata from "@/types/ProjectMetadata";
import { ChevronDownIcon } from "@heroicons/react/24/solid";
import getReleaseUrl from "@/utils/getReleaseUrl";

function ReleaseDropdown() {
  const [summary, setSummary] = useState<ReleaseSummary>({
    release: undefined,
    suites: undefined,
    suite: undefined,
    date: "",
  });

  const [currProjectMetadata, setCurrProjectMetadata] = useState<
    ProjectMetadata | undefined
  >();

  useEffect(() => {
    fetch(
      "https://raw.githubusercontent.com/stanford-crfm/helm/main/helm-frontend/project_metadata.json",
    )
      .then((response) => response.json())
      .then((data: ProjectMetadata[]) => {
        // set currProjectMetadata to val where projectMetadataEntry.id matches window.PROJECT_ID
        if (window.PROJECT_ID) {
          const currentEntry = data.find(
            (entry) => entry.id === window.PROJECT_ID,
          );
          setCurrProjectMetadata(currentEntry);
          // handles falling back to HELM lite as was previously done in this file
        } else {
          const currentEntry = data.find((entry) => entry.id === "lite");
          setCurrProjectMetadata(currentEntry);
        }
      })
      .catch((error) => {
        console.error("Error fetching JSON:", error);
      });
  }, []);

  function getReleases(): string[] {
    return currProjectMetadata !== undefined &&
      currProjectMetadata.releases !== undefined
      ? currProjectMetadata.releases
      : ["v1.0.0"];
  }

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const summ = await getReleaseSummary(controller.signal);
      setSummary(summ);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  const releases = getReleases();

  if (!summary.release && !summary.suite) {
    return null;
  }

  const releaseInfo = `Release ${summary.release || summary.suite} (${
    summary.date
  })`;

  if (releases.length <= 1) {
    return <div>{releaseInfo}</div>;
  }

  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="normal-case bg-white border-0 block whitespace-nowrap"
        aria-haspopup="true"
        aria-controls="menu"
      >
        {releaseInfo}&nbsp;
        <ChevronDownIcon
          fill="black"
          color="black"
          className="inline text w-4 h-4"
        />
      </div>
      <ul
        tabIndex={0}
        className="dropdown-content z-[1] menu p-1 shadow-lg bg-base-100 rounded-box w-max text-base"
        role="menu"
      >
        {releases.map((release) => (
          <li>
            <a
              href={getReleaseUrl(
                release,
                currProjectMetadata ? currProjectMetadata.id : "lite",
              )}
              className="block"
              role="menuitem"
            >
              {release}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ReleaseDropdown;
