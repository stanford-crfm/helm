import { useEffect, useState } from "react";
import { ChevronUpDownIcon } from "@heroicons/react/24/solid";
import type GroupsTable from "@/types/GroupsTable";
import RowValue from "@/components/RowValue";

interface Props {
  groupsTables: GroupsTable[];
  activeGroup: number;
  ignoreHref?: boolean;
  sortable?: boolean;
  sortFirstMetric?: boolean;
}

export default function GroupsTables({
  groupsTables,
  activeGroup,
  ignoreHref = false,
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

  useEffect(() => {
    setActiveGroupsTable({ ...groupsTables[activeGroup] });
  }, [activeGroup, groupsTables]);

  const handleSort = (columnIndex: number) => {
    let sort = sortDirection;
    if (activeSortColumn === columnIndex) {
      sort = sort * -1;
    } else {
      sort = 1;
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
      handleSort(activeSortColumn);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortFirstMetric, activeSortColumn]);

  return (
    <div>
      <table className="rounded-lg shadow-md table">
        <thead>
          <tr>
            {activeGroupsTable.header.map((headerValue, idx) => (
              <th
                key={`${activeGroup}-${idx}`}
                className={`${
                  idx === activeSortColumn ? "bg-gray-100 " : "bg-white"
                } ${
                  idx === 0 ? "left-0 z-10" : ""
                } whitespace-nowrap sticky top-0`}
              >
                <div className="flex gap-2 items-center">
                  <span>{headerValue.value}</span>
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
          {activeGroupsTable.rows.map((row, idx) => (
            <tr key={`${activeGroup}-${idx}`}>
              {row.map((rowValue, idx) => (
                <td
                  key={`${activeGroup}-${idx}`}
                  className={`${idx == 0 ? "text-lg sticky left-0" : ""} ${
                    activeSortColumn === idx ? "bg-gray-100" : "bg-white"
                  }`}
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
                    <RowValue
                      ignoreHref={ignoreHref && idx === 0}
                      value={rowValue}
                    />
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
