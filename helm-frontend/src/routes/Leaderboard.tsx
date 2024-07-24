import { useEffect, useState } from "react";

import PageTitle from "@/components/PageTitle";
import Loading from "@/components/Loading";
import GroupLeaderboard from "@/components/GroupLeaderboard";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";

export default function Leaderboard() {
  const [schema, setSchema] = useState<Schema | undefined>(undefined);
  const [activeRunGroupName, setActiveRunGroupName] = useState<
    string | undefined
  >(undefined);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schemaResult = await getSchema(controller.signal);
      setSchema(schemaResult);
      if (schemaResult.run_groups.length > 0) {
        setActiveRunGroupName(schemaResult.run_groups[0].name);
      }
      console.log(schemaResult.run_groups);
    }
    void fetchData();
    return () => controller.abort();
  }, []);

  if (schema && schema.run_groups.length === 0) {
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

  if (schema === undefined || activeRunGroupName === undefined) {
    return <Loading />;
  }

  return (
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
          {schema.run_groups.map((group, index) => (
            <option key={index} value={group.name}>
              {group.short_display_name || group.display_name}
            </option>
          ))}
        </select>
      </div>
      <GroupLeaderboard schema={schema} runGroupName={activeRunGroupName} />
    </div>
  );
}
