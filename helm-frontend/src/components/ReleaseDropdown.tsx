import { useEffect, useState } from "react";
import getBenchmarkRelease from "@/utils/getBenchmarkRelease";
import getReleaseSummary from "@/services/getReleaseSummary";
import ReleaseSummary from "@/types/ReleaseSummary";

function ReleaseDropdown() {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [summary, setSummary] = useState<ReleaseSummary>({
    release: "",
    suites: [],
    date: "",
  });
  const release = getBenchmarkRelease();

  function reformatDate(date: string): string {
    const [year, day, month] = date.split("-");
    const formattedDate = `${day}/${month}/${year.slice(-2)}`;
    return formattedDate;
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

  const accessibleReleases = ["v0.4.0", "v0.3.0", "v0.2.2"]; // this could also read from a config file in the future

  return (
    <div>
      <div className="inline-flex items-center">
        {/* Chevron Button */}
        <button
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className="inline-flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75"
        >
          <div> Release: {release + " - " + reformatDate(summary.date)} </div>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 ml-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>

      {dropdownOpen && (
        <div className="absolute mt-2 w-max translate-x-4 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5">
          <div
            className="py-1"
            role="menu"
            aria-orientation="vertical"
            aria-labelledby="options-menu"
          >
            {accessibleReleases.map((currRelease) => (
              <div
                className="block px-4 py-2 text-md text-gray-700 hover:bg-gray-100 hover:text-gray-900"
                role="menuitem"
              >
                <a href={"https://crfm.stanford.edu/helm/" + currRelease}>
                  <div className="flex items-center">
                    <span>{currRelease}</span>
                  </div>
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ReleaseDropdown;
