import { useEffect, useState } from "react";

import PageTitle from "@/components/PageTitle";
import Loading from "@/components/Loading";
import GroupLeaderboard from "@/components/GroupLeaderboard";
import getGroupsTables from "@/services/getGroupsTables";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";

interface GroupEntry {
  title: string;
  name: string;
}

export default function Leaderboard() {
  const [schema, setSchema] = useState<Schema | undefined>(undefined);
  const [groupEntries, setGroupEntries] = useState<GroupEntry[] | undefined>(
    undefined,
  );
  const [activeRunGroupName, setActiveRunGroupName] = useState<
    string | undefined
  >(undefined);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schemaPromise = getSchema(controller.signal);
      const groupsTablesPromise = getGroupsTables(controller.signal);

      const schemaResult = await schemaPromise;
      setSchema(schemaResult);

      const groupsTables = await groupsTablesPromise;
      const groupEntriesResult: GroupEntry[] = [];
      groupsTables.forEach((groupTable) => {
        groupTable.rows.forEach((row) => {
          groupEntriesResult.push({
            title: String(row[0].value),
            name: row[0].href.replace("?group=", ""),
          });
        });
      });
      setGroupEntries(groupEntriesResult);
    }
    void fetchData();
    return () => controller.abort();
  }, []);

  if (schema === undefined || groupEntries === undefined) {
    return <Loading />;
  }

  if (groupEntries.length === 0) {
    return (
      <>
        <PageTitle
          title={"HELM Leaderboard"}
          subtitle={
            "The HELM leaderboard shows how the various models perform across different scenarios and metrics."
          }
          markdown={true}
        />
        <div className="divider"></div>
        <p className="text-center mt-8">Group currently has no results.</p>
      </>
    );
  }

  const runGroupName =
    activeRunGroupName !== undefined
      ? activeRunGroupName
      : groupEntries[0].name;

  return (
    <>
      <div className="flex flex-row justify-between">
        <PageTitle
          title={"HELM Leaderboard"}
          subtitle={
            "The HELM leaderboard shows how the various models perform across different scenarios and metrics."
          }
          markdown={true}
        />
        <div className="w-64 pt-8">
          <label
            htmlFor="group"
            className="block text-sm font-medium text-gray-700"
          >
            Select a group:
          </label>
          <select
            id="group"
            name="group"
            onChange={(e) => setActiveRunGroupName(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring focus:border-blue-300 rounded-md"
          >
            {groupEntries.map((group, index) => (
              <option key={index} value={group.name}>
                {group.title}
              </option>
            ))}
          </select>
        </div>
      </div>
      <GroupLeaderboard
        key={runGroupName}
        schema={schema}
        runGroupName={runGroupName}
      />
    </>
  );
}
