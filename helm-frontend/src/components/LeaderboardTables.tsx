/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import { ChevronUpDownIcon } from "@heroicons/react/24/solid";
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
  filtered?: boolean;
  modelsToFilter?: string[];
  numModelsToAutoFilter?: number;
  filteredCols?: any[];
}

export default function LeaderboardTables({
  groupsTables,
  activeGroup,
  sortable = true,
  sortFirstMetric = true,
}: Props) {
  const [activeSortColumn, setActiveSortColumn] = useState<number | undefined>(
    sortFirstMetric ? 1 : undefined,
  );
  const [activeGroupsTable, setActiveGroupsTable] = useState<GroupsTable>({
    ...groupsTables[activeGroup],
  });
  const [sortDirection, setSortDirection] = useState<number>(1);

  function truncateHeader(value: string): string {
    if (value.length > 30) {
      return value.substring(0, 27) + "...";
    }
    return value;
  }

  // TODO remove truncation once a visually suitable version of wrapping is determined
  const getHeaderValue = (headerValueObject: HeaderValue): string => {
    const stringsToIgnore = ["AIRBench 2024 -", "-book"];
    if (headerValueObject.value === "Model/adapter") {
      return "Model";
      // hardcoded values to remove
    } else if (
      stringsToIgnore.some((str) => headerValueObject.value.includes(str))
    ) {
      let updatedValue = headerValueObject.value;
      stringsToIgnore.forEach((str) => {
        updatedValue = updatedValue.replace(str, "");
      });
      return truncateHeader(updatedValue);
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
  // create delimiter to parse out run group name (need to replace hyphen as some run names have hyphens in them)
  function replaceLastHyphen(str: string): string {
    const lastIndex = str.lastIndexOf(" - ");
    if (lastIndex === -1) {
      return str;
    }
    return str.substring(0, lastIndex) + "*" + str.substring(lastIndex + 1);
  }
  const getGroupForRunName = (rawGroup: string): string => {
    const groupSplit = replaceLastHyphen(rawGroup).split("*");
    const group = groupSplit[0].trim();
    if (schema) {
      const foundItem = schema.run_groups.find(
        (item) =>
          item.display_name === group || item.short_display_name === group,
      );
      if (foundItem) {
        return foundItem.name;
      }
    }
    return "";
  };

  useEffect(() => {
    setActiveGroupsTable({ ...groupsTables[activeGroup] });
  }, [activeGroup, groupsTables]);

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
      <div>
        <div>
          <table className="rounded-lg shadow-md table">
            <thead>
              <tr>
                {activeGroupsTable.header.map(
                  (headerValue: HeaderValue, idx) => (
                    <th
                      key={`${activeGroup}-${idx}`}
                      className={`${
                        idx === activeSortColumn ? "bg-gray-100" : "bg-white"
                      } ${idx === 0 ? "left-0 z-40" : ""} ${
                        headerValue.description
                          ? "underline decoration-dashed decoration-gray-300	"
                          : ""
                      }  whitespace-nowrap px-4 sticky top-0`}
                      title={
                        headerValue.description ? headerValue.description : ""
                      }
                    >
                      <div className="z-20 flex justify-between items-center min-w-48 w-48 max-w-48 text-wrap">
                        <span className={`inline-block w-full break-words`}>
                          {getHeaderValue(headerValue)}
                        </span>

                        {sortable ? (
                          <button
                            className="link"
                            onClick={() =>
                              handleSort(idx, headerValue.lower_is_better)
                            }
                          >
                            <ChevronUpDownIcon className="w-6 h-6" />
                          </button>
                        ) : null}
                      </div>
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {activeGroupsTable.rows.map((row, idx) => (
                <tr key={`${activeGroup}-${idx}`}>
                  {row.map((rowValue, cellIdx) => (
                    <td
                      key={`${activeGroup}-${cellIdx}`}
                      className={`${
                        cellIdx === 0 ? "z-20 text-lg sticky left-0" : "z-0"
                      } ${idx % 2 === 0 ? "bg-gray-50" : "bg-white"}`}
                    >
                      {cellIdx == 1 ? (
                        <div
                          className={`${
                            rowValue &&
                            rowValue.style &&
                            rowValue.style["font-weight"] &&
                            rowValue.style["font-weight"] === "bold"
                              ? "font-bold"
                              : ""
                          }`}
                        >
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
                        </div>
                      ) : (
                        <div
                          className={`${
                            rowValue &&
                            rowValue.style &&
                            rowValue.style["font-weight"] &&
                            rowValue.style["font-weight"] === "bold"
                              ? "font-bold"
                              : ""
                          } ${
                            cellIdx === 0
                              ? "underline decoration-dashed decoration-gray-300 z-10"
                              : "z-0"
                          }`}
                        >
                          <RowValue
                            value={{ ...rowValue }}
                            title={
                              String(row[0].value) === rowValue.value
                                ? getModelDesc(String(row[0].value))
                                : `Click value to see predictions for ${getGroupForRunName(
                                    getHeaderValue(
                                      activeGroupsTable.header[cellIdx],
                                    ),
                                  )}: ${getModelForRunName(
                                    String(row[0].value),
                                  )}`
                            }
                          />
                        </div>
                      )}
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
