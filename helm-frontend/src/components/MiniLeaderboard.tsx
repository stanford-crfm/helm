import { useEffect, useState } from "react";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import MiniLeaderboardTables from "@/components/MiniLeaderboardTables";
import type GroupsTable from "@/types/GroupsTable";
import type GroupMetadata from "@/types/GroupMetadata";
import getGroupsTablesByName from "@/services/getGroupTablesByName";
import getGroupsMetadata from "@/services/getGroupsMetadata";
import Loading from "@/components/Loading";

interface Props {
  runGroupName?: string;
  tableIndexToDisplay?: number;
  numRowsToDisplay?: number;
  sortColumnIndex?: number;
  displayColumnIndexes?: number[];
}

export default function MiniLeaderboard({
  runGroupName = undefined,
  tableIndexToDisplay = 0,
  numRowsToDisplay = 10,
  sortColumnIndex = 1,
  displayColumnIndexes = [0, 1],
}: Props) {
  const [schema, setSchema] = useState<Schema | undefined>(undefined);
  const [groupTable, setGroupTable] = useState<GroupsTable | undefined>(
    undefined,
  );
  const [groupMetadata, setGroupMetadata] = useState<
    GroupMetadata | undefined
  >();

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const groupsMetadataPromise = getGroupsMetadata(controller.signal);
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
      const groupsMetadataResult = await groupsMetadataPromise;
      setGroupMetadata(groupsMetadataResult[selectedGroupName]);
    }
    void fetchData();
    return () => controller.abort();
  }, [runGroupName, tableIndexToDisplay]);

  if (
    schema === undefined ||
    groupTable === undefined ||
    groupMetadata === undefined
  ) {
    return <Loading />;
  }

  return (
    <MiniLeaderboardTables
      schema={schema}
      groupTable={groupTable}
      numRowsToDisplay={numRowsToDisplay}
      sortColumnIndex={sortColumnIndex}
      displayColumnIndexes={displayColumnIndexes}
    />
  );
}
