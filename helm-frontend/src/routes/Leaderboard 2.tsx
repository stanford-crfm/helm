import { useEffect, useState } from "react";

import PageTitle from "@/components/PageTitle";
import Loading from "@/components/Loading";
import GroupLeaderboard from "@/components/GroupLeaderboard";
import getGroupsTables from "@/services/getGroupsTables";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import { useParams, useNavigate } from "react-router-dom";
import RunGroup from "@/types/RunGroup";

interface GroupEntry {
  title: string;
  name: string;
}

interface HeaderToGroupEntries {
  [Key: string]: GroupEntry[];
}

export default function Leaderboard() {
  const { groupName } = useParams();
  const navigate = useNavigate();

  const [schema, setSchema] = useState<Schema | undefined>(undefined);
  const [headerToGroupEntries, setHeaderToGroupEntries] = useState<
    HeaderToGroupEntries | undefined
  >();
  const [defaultGroupName, setDefaultGroupName] = useState<
    string | undefined
  >();

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schemaPromise = getSchema(controller.signal);
      const groupsTablesPromise = getGroupsTables(controller.signal);

      const schemaResult = await schemaPromise;
      setSchema(schemaResult);
      let defaultGroupNameResult: string | undefined;
      const groupsTables = await groupsTablesPromise;
      const headerToGroupEntriesResult: HeaderToGroupEntries = {};
      groupsTables.forEach((groupTable) => {
        headerToGroupEntriesResult[groupTable.title] = [];
        groupTable.rows.forEach((row) => {
          const groupName = row[0].href.replace("?group=", "");
          if (defaultGroupNameResult === undefined) {
            defaultGroupNameResult = groupName;
          }
          headerToGroupEntriesResult[groupTable.title].push({
            title: String(row[0].value),
            name: groupName,
          });
        });
      });
      setHeaderToGroupEntries(headerToGroupEntriesResult);
      setDefaultGroupName(defaultGroupNameResult);
    }
    void fetchData();
    return () => controller.abort();
  }, []);

  const runGroupName = groupName || defaultGroupName;
  if (
    schema === undefined ||
    headerToGroupEntries === undefined ||
    runGroupName === undefined
  ) {
    return <Loading />;
  }

  let groupMetadata: RunGroup | undefined;
  for (const runGroup of schema.run_groups) {
    if (runGroup.name === runGroupName) {
      groupMetadata = runGroup;
    }
  }

  return (
    <>
      <div className="flex flex-row">
        <div className="w-3/4">
          {groupMetadata ? (
            <PageTitle
              title={"Leaderboard: " + groupMetadata.display_name}
              subtitle={groupMetadata.description}
              markdown={true}
            />
          ) : (
            <PageTitle
              title={"Leaderboard"}
              subtitle={
                "The HELM leaderboard shows how the various models perform across different scenarios and metrics."
              }
              markdown={true}
            />
          )}
        </div>

        <div className="w-1/4 pt-8">
          <label
            htmlFor="group"
            className="block text-sm font-medium text-gray-700"
          >
            Select a group:
            <select
              id="group"
              name="group"
              onChange={(e) => navigate("/leaderboard/" + e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring focus:border-blue-300 rounded-md"
              value={runGroupName}
            >
              {Object.entries(headerToGroupEntries).map(
                ([header, groupEntries]) => (
                  <optgroup key={header} label={header}>
                    {groupEntries.map((groupEntry) => (
                      <option key={groupEntry.name} value={groupEntry.name}>
                        {groupEntry.title}
                      </option>
                    ))}
                  </optgroup>
                ),
              )}
            </select>
          </label>
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
