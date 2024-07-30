import { useEffect, useState } from "react";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import LeaderboardTable from "@/components/LeaderboardTable";
import type GroupsTable from "@/types/GroupsTable";
import getGroupsTablesByName from "@/services/getGroupTablesByName";
import Loading from "@/components/Loading";

interface Props {
  runGroupName?: string;
  tableIndexToDisplay?: number;
  numRowsToDisplay?: number;
  sortColumnIndex?: number;
}

export default function MiniLeaderboard({
  runGroupName = undefined,
  tableIndexToDisplay = 0,
  numRowsToDisplay = 10,
  sortColumnIndex = 1,
}: Props) {
  const [schema, setSchema] = useState<Schema | undefined>(undefined);
  const [groupTable, setGroupTable] = useState<GroupsTable | undefined>(
    undefined,
  );

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schemaResult = await getSchema(controller.signal);
      setSchema(schemaResult);
      const runGroups = schemaResult.run_groups;
      if (runGroups.length === 0) {
        return;
      }
      const selectedGroupName = runGroupName || runGroups[0].name;
      const groupTablesResult = await getGroupsTablesByName(
        selectedGroupName,
        controller.signal,
      );
      setGroupTable(groupTablesResult[tableIndexToDisplay]);
    }
    void fetchData();
    return () => controller.abort();
  }, [runGroupName, tableIndexToDisplay]);

  if (schema === undefined || groupTable === undefined) {
    return <Loading />;
  }

  return (
    <div
      className="rounded-2xl overflow-hidden border-2 bg-white p-1 mx-2 my-0 overflow-x-auto"
      style={{ overflow: "auto", justifyContent: "space-between" }}
    >
      <LeaderboardTable
        schema={schema}
        groupTable={groupTable}
        numRowsToDisplay={numRowsToDisplay}
        sortColumnIndex={sortColumnIndex}
        displayColumnIndexes={[0, 1]}
        sortable={false}
        miniStyle={true}
      />
    </div>
  );
}
