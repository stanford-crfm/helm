import { useEffect, useState } from "react";
import { useParams } from "react-router";
import PageTitle from "@/components/PageTitle";
import Loading from "@/components/Loading";
import GroupLeaderboard from "@/components/GroupLeaderboard";
import getSchema from "@/services/getSchema";
import Schema from "@/types/Schema";
import RunGroup from "@/types/RunGroup";

export default function Group() {
  const { groupName } = useParams();
  const [schema, setSchema] = useState<Schema | undefined>(undefined);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schemaPromise = getSchema(controller.signal);
      const schemaResult = await schemaPromise;
      setSchema(schemaResult);
    }
    void fetchData();
    return () => controller.abort();
  }, []);

  const getGroupMetadata = (): RunGroup | undefined => {
    if (schema !== undefined) {
      for (const runGroup of schema.run_groups) {
        if (runGroup.name === groupName) {
          return runGroup;
        }
      }
    }
    return undefined;
  };

  const groupMetadata: RunGroup | undefined = getGroupMetadata();

  if (schema === undefined) {
    return <Loading />;
  }

  if (groupMetadata === undefined) {
    return <div>Group "{groupName}" not found.</div>;
  }

  return (
    <>
      <PageTitle
        title={groupMetadata.display_name}
        subtitle={groupMetadata.description}
        markdown={true}
        className="mr-8"
      />
      <GroupLeaderboard
        key={groupMetadata.name}
        schema={schema}
        runGroupName={groupMetadata.name}
      />
    </>
  );
}
