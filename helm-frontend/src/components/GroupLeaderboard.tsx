import { useEffect, useState } from "react";
import type Schema from "@/types/Schema";
import Tab from "@/components/Tab";
import Tabs from "@/components/Tabs";
import LeaderboardTable from "@/components/LeaderboardTable";
import TransposedLeaderboardTable from "@/components/TransposedLeaderboardTable";
import type GroupsTable from "@/types/GroupsTable";
import getGroupsTablesByName from "@/services/getGroupTablesByName";
import Loading from "@/components/Loading";

interface Props {
  schema: Schema;
  runGroupName: string;
  numRowsToDisplay?: number;
  sortColumnIndex?: number;
  displayColumnIndexes?: number[];
}

export default function GroupLeaderboard({
  schema,
  runGroupName,
  numRowsToDisplay = -1,
}: Props) {
  const [groupTables, setGroupTables] = useState<GroupsTable[] | undefined>();
  const [tableIndex, setTableIndex] = useState<number>(0);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const groupTablesResult = await getGroupsTablesByName(
        runGroupName,
        controller.signal,
      );
      setGroupTables(groupTablesResult);
    }
    void fetchData();
    return () => controller.abort();
  }, [schema, runGroupName]);

  if (groupTables === undefined || groupTables.length === 0) {
    return <Loading />;
  }

  if (groupTables.length === 0) {
    return <div>Group currently has no tables.</div>;
  }

  // TODO: Add an flag to the schema for transposing tables.
  const isMedHELM = schema.run_groups[0].name == "medhelm_scenarios";

  return (
    <div>
      {groupTables.length > 1 ? (
        <Tabs>
          {groupTables.map((groupTable, idx) => (
            <Tab
              key={idx}
              active={idx === tableIndex}
              onClick={() => setTableIndex(idx)}
            >
              {groupTable.title}
            </Tab>
          ))}
        </Tabs>
      ) : null}
      {isMedHELM ? (
        <TransposedLeaderboardTable
          key={`${runGroupName}-${tableIndex}`}
          schema={schema}
          groupTable={groupTables[tableIndex]}
          numRowsToDisplay={numRowsToDisplay}
          sortColumnIndex={1}
        />
      ) : (
        <LeaderboardTable
          key={`${runGroupName}-${tableIndex}`}
          schema={schema}
          groupTable={groupTables[tableIndex]}
          numRowsToDisplay={numRowsToDisplay}
          sortColumnIndex={1}
        />
      )}
    </div>
  );
}
