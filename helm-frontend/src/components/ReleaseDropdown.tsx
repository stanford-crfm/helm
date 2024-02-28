import { useEffect, useState } from "react";
import getReleaseSummary from "@/services/getReleaseSummary";
import ReleaseSummary from "@/types/ReleaseSummary";
import { ChevronDownIcon } from "@heroicons/react/24/solid";

function getReleases(): string[] {
  // TODO: Fetch this from a configuration file.
  if (window.HELM_TYPE === "LITE") {
    return ["v1.1.0", "v1.0.0"];
  } else if (window.HELM_TYPE === "CLASSIC") {
    return ["v0.4.0", "v0.3.0", "v0.2.4", "v0.2.3", "v0.2.2"];
  } else if (window.HELM_TYPE === "HEIM") {
    return ["v1.1.0", "v1.0.0"];
  } else if (window.HELM_TYPE === "INSTRUCT") {
    return ["v1.0.0"];
  }
  return ["v1.1.0", "v1.0.0"];
}

function getReleaseUrl(version: string): string {
  // TODO: Fetch this from a configuration file.
  if (window.HELM_TYPE === "LITE") {
    return `https://crfm.stanford.edu/helm/lite/${version}/`;
  } else if (window.HELM_TYPE === "CLASSIC") {
    return `https://crfm.stanford.edu/helm/classic/${version}/`;
  } else if (window.HELM_TYPE === "HEIM") {
    return `https://crfm.stanford.edu/heim/${version}/`;
  } else if (window.HELM_TYPE === "INSTRUCT") {
    return `https://crfm.stanford.edu/helm/instruct/${version}/`;
  }
  return `https://crfm.stanford.edu/helm/lite/${version}/`;
}

function ReleaseDropdown() {
  const [summary, setSummary] = useState<ReleaseSummary>({
    release: undefined,
    suites: undefined,
    suite: undefined,
    date: "",
  });
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

  const releaseInfo = `Release ${
    summary.release || summary.suite || "unknown"
  } (${summary.date})`;

  if (releases.length <= 1) {
    return <div>{releaseInfo}</div>;
  }

  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="normal-case bg-white border-0"
        aria-haspopup="true"
        aria-controls="menu"
      >
        {releaseInfo}{" "}
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
            <a href={getReleaseUrl(release)} className="block" role="menuitem">
              {release}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ReleaseDropdown;
