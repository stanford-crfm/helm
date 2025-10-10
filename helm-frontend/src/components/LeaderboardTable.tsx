import { useState } from "react";
import { ChevronUpDownIcon } from "@heroicons/react/24/solid";
import type GroupsTable from "@/types/GroupsTable";
import RowValue from "@/components/RowValue";
import Schema from "@/types/Schema";
import RowValueType from "@/types/RowValue";
import HeaderValue from "@/types/HeaderValue";

interface Props {
  schema: Schema;
  groupTable: GroupsTable;
  numRowsToDisplay: number;
  sortColumnIndex?: number;
  sortable?: boolean;
  displayColumnIndexes?: number[] | undefined;
  miniStyle?: boolean;
}

export default function LeaderboardTable({
  schema,
  groupTable,
  numRowsToDisplay,
  sortColumnIndex = 1,
  sortable = true,
  displayColumnIndexes = undefined,
  miniStyle = false,
}: Props) {
  const [sortDirection, setSortDirection] = useState<number>(1);
  const [selectedColumnIndex, setSelectedColumnIndex] = useState<number>(
    Math.min(groupTable.header.length - 1, sortColumnIndex),
  );

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

  const handleSort = (columnIndex: number) => {
    // If the column is already selected, just reverse the direction.
    if (columnIndex === selectedColumnIndex) {
      setSortDirection(sortDirection * -1);
    } else {
      // Special-case sorting by model name (i.e. first column)
      setSortDirection(columnIndex === 0 ? -1 : 1);
    }
    setSelectedColumnIndex(columnIndex);
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

  const getSortedRows = (): RowValueType[][] => {
    const lowerIsBetter =
      groupTable.header[selectedColumnIndex].lower_is_better;
    const sortSign = sortDirection * (lowerIsBetter ? 1 : -1);
    const rows = groupTable.rows.slice();
    rows.sort((a, b) => {
      const av = a[selectedColumnIndex]?.value;
      const bv = b[selectedColumnIndex]?.value;
      if (av !== undefined && bv === undefined) {
        return -1;
      }
      if (bv !== undefined && av === undefined) {
        return 1;
      }
      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * sortSign;
      }
      if (typeof av === "string" && typeof bv === "string") {
        if (sortSign === 1) {
          return av.localeCompare(bv);
        }
        return bv.localeCompare(av);
      }

      return 0;
    });
    return numRowsToDisplay > 0 ? rows.slice(0, numRowsToDisplay) : rows;
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

  return (
    <table
      className={miniStyle ? "table w-full" : "rounded-lg shadow-md table"}
    >
      <thead>
        <tr>
          {groupTable.header
            .filter(
              (_, cellIdx) =>
                displayColumnIndexes === undefined ||
                displayColumnIndexes.includes(cellIdx),
            )
            .map((headerValue: HeaderValue, idx) => (
              <th
                key={`$${idx}`}
                className={`${
                  idx === selectedColumnIndex ? "bg-gray-100" : "bg-white"
                } ${idx === 0 ? "left-0 z-40" : ""} ${
                  headerValue.description
                    ? "underline decoration-dashed decoration-gray-300	"
                    : ""
                }  whitespace-nowrap px-4 sticky top-0`}
                title={headerValue.description ? headerValue.description : ""}
              >
                <div
                  className={
                    miniStyle
                      ? "flex gap-2 items-center"
                      : "z-20 flex justify-between items-center min-w-48 w-48 max-w-48 text-wrap"
                  }
                >
                  <span className={`inline-block w-full break-words`}>
                    {getHeaderValue(headerValue)}
                  </span>

                  {sortable ? (
                    <button className="link" onClick={() => handleSort(idx)}>
                      <ChevronUpDownIcon className="w-6 h-6" />
                    </button>
                  ) : null}
                </div>
              </th>
            ))}
        </tr>
      </thead>
      <tbody>
        {getSortedRows().map((row, idx) => (
          <tr key={`$${row[0].value}`}>
            {row
              .filter(
                (_, cellIdx) =>
                  displayColumnIndexes === undefined ||
                  displayColumnIndexes.includes(cellIdx),
              )
              .map((rowValue, cellIdx) => (
                <td
                  key={`${cellIdx}`}
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
                            : `Click value to see predictions for ${String(
                                row[0].value,
                              )} for ${getGroupForRunName(
                                getHeaderValue(groupTable.header[cellIdx]),
                              )}: ${getModelForRunName(String(row[0].value))}`
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
  );
}
