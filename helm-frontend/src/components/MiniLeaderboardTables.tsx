/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import type GroupsTable from "@/types/GroupsTable";
import RowValue from "@/components/RowValue";
import Schema from "@/types/Schema";
import HeaderValue from "@/types/HeaderValue";
import getSchema from "@/services/getSchema";

interface Props {
  groupsTables: GroupsTable[];
  activeGroup: number;
  ignoreHref?: boolean;
  sortable?: boolean;
  sortFirstMetric?: boolean;
  modelsToFilter?: string[];
  numModelsToAutoFilter?: number;
  filteredCols?: any[];
}

export default function MiniLeaderboardTables({
  groupsTables,
  activeGroup,
  sortFirstMetric = true,
  filteredCols = [],
  modelsToFilter = [],
  numModelsToAutoFilter = 0, // if non-zero, sets how many models to filter down to (ranked by first column)
}: Props) {
  const [activeSortColumn, setActiveSortColumn] = useState<number | undefined>(
    sortFirstMetric ? 1 : undefined,
  );
  const [activeGroupsTable, setActiveGroupsTable] = useState<GroupsTable>({
    ...groupsTables[activeGroup],
  });
  const [sortDirection, setSortDirection] = useState<number>(1);
  const [filteredModels, setFilteredModels] =
    useState<string[]>(modelsToFilter);

  function truncateHeader(value: string): string {
    if (value.length > 30) {
      return value.substring(0, 27) + "...";
    }
    return value;
  }

  // TODO remove truncation once a visually suitable version of wrapping is determined
  const getHeaderValue = (headerValueObject: HeaderValue): string => {
    if (headerValueObject.value === "Model/adapter") {
      return "Model";
    } else if (headerValueObject.value.includes("-book")) {
      return truncateHeader(headerValueObject.value.replace("-book", ""));
    } else {
      return truncateHeader(headerValueObject.value);
    }
  };

  const [schema, setSchema] = useState<Schema | undefined>(undefined);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schema = await getSchema(controller.signal);
      setSchema(schema);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  const getModelDesc = (model: string): string => {
    if (schema) {
      const foundItem = schema.models.find(
        (item) => item.display_name === model,
      );
      if (foundItem) {
        let toRet = foundItem.description;
        if (toRet.includes("/")) {
          toRet = toRet.replace("/", "_");
          return toRet;
        } else {
          return toRet;
        }
      }
    }
    return "";
  };

  const getModelForRunName = (model: string): string => {
    if (schema) {
      const foundItem = schema.models.find(
        (item) => item.display_name === model,
      );
      if (foundItem) {
        let toRet = foundItem.name;
        if (toRet.includes("/")) {
          toRet = toRet.replace("/", "_");
          return toRet;
        } else {
          return toRet;
        }
      }
    }
    return "";
  };

  useEffect(() => {
    setActiveGroupsTable({ ...groupsTables[activeGroup] });
    // upon receiving and setting data for current table, use sort to figure out n top models
    const lowerIsBetter = activeGroupsTable.header[1].lower_is_better;
    if (numModelsToAutoFilter) {
      const activeRows = groupsTables[0].rows;
      const sortedRows = activeRows.sort((a, b) => {
        // assumes we sort by column 1, which represents Mean Win Rate in the Core Scenarios table
        // this assumption works as numModelsToAutoFilter is only used in mini leaderboards
        // which always have one main scenario we sort by
        return lowerIsBetter
          ? Number(a[1].value) - Number(b[1].value)
          : Number(b[1].value) - Number(a[1].value);
      });
      // Get the top ModelsToAutoFilter
      const topNumRows = sortedRows.slice(0, numModelsToAutoFilter);
      const topNumRowNames = topNumRows.map((row) => String(row[0].value));
      setFilteredModels(topNumRowNames);
    }
  }, [activeGroup, activeGroupsTable, groupsTables, numModelsToAutoFilter]);

  const handleSort = (columnIndex: number, lowerIsBetter: boolean = false) => {
    let sort = sortDirection;
    if (activeSortColumn === columnIndex) {
      sort = sort * -1;
    } else {
      sort = 1;
    }

    if (lowerIsBetter) {
      sort = sort * -1;
    }

    setActiveSortColumn(columnIndex);
    setSortDirection(sort);

    setActiveGroupsTable((prev) => {
      const group = { ...prev };
      group.rows.sort((a, b) => {
        const av = a[columnIndex]?.value;
        const bv = b[columnIndex]?.value;
        if (av !== undefined && bv === undefined) {
          return -1;
        }
        if (bv !== undefined && av === undefined) {
          return 1;
        }
        if (typeof av === "number" && typeof bv === "number") {
          return (av - bv) * sort;
        }
        if (typeof av === "string" && typeof bv === "string") {
          if (sort === 1) {
            return av.localeCompare(bv);
          }
          return bv.localeCompare(av);
        }

        return 0;
      });

      return group;
    });
  };
  useEffect(() => {
    if (sortFirstMetric && activeSortColumn) {
      handleSort(
        activeSortColumn,
        activeGroupsTable.header[activeSortColumn].lower_is_better,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortFirstMetric, activeSortColumn]);

  return (
    <>
      <div
        className="rounded-2xl overflow-hidden border-2 bg-white p-1 mx-2 my-0"
        style={{ overflow: "auto", justifyContent: "space-between" }}
      >
        <div className="overflow-x-auto">
          <table className="table w-full">
            <thead>
              <tr>
                {activeGroupsTable.header
                  .filter(
                    (_, cellIdx) =>
                      filteredCols.length === 0 ||
                      filteredCols.includes(cellIdx),
                  )
                  .map((headerValue, idx) => (
                    <th
                      key={`${activeGroup}-${idx}`}
                      className={`${
                        idx === activeSortColumn ? "bg-gray-100" : ""
                      } ${
                        headerValue.description
                          ? "underline decoration-dashed"
                          : ""
                      } whitespace-nowrap px-4 `}
                      title={
                        headerValue.description ? headerValue.description : ""
                      }
                    >
                      <div className="flex gap-2 items-center">
                        <span>{getHeaderValue(headerValue)}</span>
                      </div>
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {activeGroupsTable.rows
                .filter((row) => filteredModels.includes(String(row[0].value)))
                .map((row, idx) => (
                  <tr
                    key={`${activeGroup}-${idx}`}
                    className={`${idx % 2 === 0 ? "bg-gray-50" : ""}`}
                  >
                    {row
                      .filter(
                        (_, cellIdx) =>
                          filteredCols.length === 0 ||
                          filteredCols.includes(cellIdx),
                      )
                      .map((rowValue, cellIdx) => (
                        <td
                          key={`${activeGroup}-${cellIdx}`}
                          className={`${cellIdx === 0 ? "text-lg" : ""}`}
                        >
                          <div
                            className={
                              rowValue &&
                              rowValue.style &&
                              rowValue.style["font-weight"] &&
                              rowValue.style["font-weight"] === "bold"
                                ? "font-bold"
                                : ""
                            }
                          >
                            {cellIdx === 0 ? (
                              <RowValue
                                value={{
                                  ...rowValue,
                                }}
                                title={getModelDesc(String(row[0].value))}
                                hideIcon
                              />
                            ) : (
                              <RowValue
                                value={{
                                  ...rowValue,
                                  href:
                                    "/runs/?q=" +
                                    getModelForRunName(String(row[0].value)),
                                }}
                                title={`Click value to see all predictions for: ${getModelForRunName(
                                  String(row[0].value),
                                )}`}
                              />
                            )}
                          </div>
                        </td>
                      ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
  // TODO: Remove unnecessary divs
}
