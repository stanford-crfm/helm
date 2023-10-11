import { SyntheticEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import getRunSpecs from "@/services/getRunSpecs";
import type RunSpec from "@/types/RunSpec";
import PageTitle from "@/components/PageTitle/PageTitle";
import Link from "@/components/Link";
import Loading from "@/components/Loading";
import Pagination from "@/components/Pagination";
import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";

const PAGE_SIZE = 100;

export default function Runs() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [runSpecs, setRunSpecs] = useState<RunSpec[]>([]);
  const [currentPage, setCurrentPage] = useState<number>(
    Number(searchParams.get("page") || 1),
  );
  const [totalPages, setTotalPages] = useState<number>(1);
  const [filteredRunSpecs, setFilteredRunSpecs] = useState<RunSpec[]>([]);
  const [useRegex, setUseRegex] = useState<boolean>(false);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const runSpecs = await getRunSpecs(controller.signal);
      setRunSpecs(runSpecs);
      setFilteredRunSpecs(runSpecs);
      setTotalPages(Math.ceil(runSpecs.length / PAGE_SIZE));
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  if (runSpecs.length === 0) {
    return <Loading />;
  }

  const handleSearch = (e: SyntheticEvent) => {
    e.preventDefault();

    const target = e.target as typeof e.target & {
      q: { value: string };
    };
    const q = (() => {
      if (target.q === undefined) {
        return "";
      }
      if (useRegex) {
        return new RegExp(target.q.value);
      }
      return target.q.value;
    })();

    const filteredRunSpecs = (() => {
      if (q === "") {
        return runSpecs;
      }
      return runSpecs.filter((runSpec) => {
        if (q instanceof RegExp) {
          return q.test(runSpec.name);
        }
        return runSpec.name.includes(q);
      });
    })();

    setFilteredRunSpecs(filteredRunSpecs);
    setTotalPages(Math.ceil(filteredRunSpecs.length / PAGE_SIZE));
  };

  const pagedRunSpecs = filteredRunSpecs.slice(
    (currentPage - 1) * PAGE_SIZE,
    (currentPage - 1) * PAGE_SIZE + PAGE_SIZE,
  );

  return (
    <>
      <PageTitle title="Runs" subtitle="All benchmark runs" />
      <form className="flex mb-8" onSubmit={handleSearch}>
        <div className="form-control">
          <input
            type="text"
            name="q"
            placeholder="Search"
            className="input input-bordered"
          />
          <label className="label">
            <span className="label-text-alt flex item-center">
              <input
                type="checkbox"
                className="toggle toggle-xs"
                checked={useRegex}
                onChange={() => setUseRegex(!useRegex)}
              />
              <span className="ml-2">Regex</span>
            </span>
            <span className="label-text-alt">
              {`${filteredRunSpecs.length} results`}
            </span>
          </label>
        </div>
        <div className="form-control ml-4">
          <button className="btn">
            <MagnifyingGlassIcon className="w-6 h-6" />
          </button>
        </div>
      </form>

      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Model</th>
              <th>Groups</th>
              <th>Adapter method</th>
              <th>Subject / Task</th>
            </tr>
          </thead>
          <tbody>
            {pagedRunSpecs.map((runSpec, idx) => (
              <tr key={`${runSpec.name}-${idx}`}>
                <td>
                  <Link to={`/runs/${runSpec.name}`}>{runSpec.name}</Link>
                </td>
                <td>{runSpec.adapter_spec.model}</td>
                <td>{runSpec.groups.join(", ")}</td>
                <td>{runSpec.adapter_spec.method}</td>
                <td>
                  {runSpec.scenario_spec.args.subject ||
                    runSpec.scenario_spec.args.task ||
                    "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 0 ? (
        <Pagination
          className="flex justify-center my-8"
          onNextPage={() => {
            const nextPage = Math.min(currentPage + 1, totalPages);
            setCurrentPage(nextPage);
            searchParams.set("page", String(nextPage));
            setSearchParams(searchParams);
          }}
          onPrevPage={() => {
            const prevPage = Math.max(currentPage - 1, 1);
            setCurrentPage(prevPage);
            searchParams.set("page", String(prevPage));
            setSearchParams(searchParams);
          }}
          currentPage={currentPage}
          totalPages={totalPages}
        />
      ) : (
        <div className="my-8 text-center">No results</div>
      )}
    </>
  );
}
