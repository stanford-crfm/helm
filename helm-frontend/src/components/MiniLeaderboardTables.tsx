import type GroupsTable from "@/types/GroupsTable";
import RowValue from "@/components/RowValue";
import Schema from "@/types/Schema";
import RowValueType from "@/types/RowValue";
import HeaderValue from "@/types/HeaderValue";

interface Props {
  schema: Schema;
  groupTable: GroupsTable;
  numRowsToDisplay: number;
  sortColumnIndex: number;
  displayColumnIndexes: number[];
}

export default function MiniLeaderboardTables({
  schema,
  groupTable,
  numRowsToDisplay,
  sortColumnIndex,
  displayColumnIndexes,
}: Props) {
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

  const getSortedRows = (): RowValueType[][] => {
    const lowerIsBetter = groupTable.header[sortColumnIndex].lower_is_better;
    const sortSign = lowerIsBetter ? 1 : -1;
    const rows = groupTable.rows.slice();
    rows.sort((a, b) => {
      const av = a[sortColumnIndex]?.value;
      const bv = b[sortColumnIndex]?.value;
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

  return (
    <div
      className="rounded-2xl overflow-hidden border-2 bg-white p-1 mx-2 my-0 overflow-x-auto"
      style={{ overflow: "auto", justifyContent: "space-between" }}
    >
      <table className="table w-full">
        <thead>
          <tr>
            {groupTable.header
              .filter(
                (_, cellIdx) =>
                  displayColumnIndexes.length === 0 ||
                  displayColumnIndexes.includes(cellIdx),
              )
              .map((headerValue, idx) => (
                <th
                  key={`${idx}`}
                  className={`${idx === sortColumnIndex ? "bg-gray-100" : ""} ${
                    headerValue.description ? "underline decoration-dashed" : ""
                  } whitespace-nowrap px-4 `}
                  title={headerValue.description ? headerValue.description : ""}
                >
                  <div className="flex gap-2 items-center">
                    <span>{getHeaderValue(headerValue)}</span>
                  </div>
                </th>
              ))}
          </tr>
        </thead>
        <tbody>
          {getSortedRows().map((row, idx) => (
            <tr
              key={`${idx}`}
              className={`${idx % 2 === 0 ? "bg-gray-50" : ""}`}
            >
              {row
                .filter(
                  (_, cellIdx) =>
                    displayColumnIndexes.length === 0 ||
                    displayColumnIndexes.includes(cellIdx),
                )
                .map((rowValue, cellIdx) => (
                  <td
                    key={`${cellIdx}`}
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
  );
}
