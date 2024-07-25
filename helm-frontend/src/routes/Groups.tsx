import { useEffect, useState } from "react";
import type GroupsTable from "@/types/GroupsTable";
import getGroupsTables from "@/services/getGroupsTables";
import PageTitle from "@/components/PageTitle";
import Tabs from "@/components/Tabs";
import Tab from "@/components/Tab";
import Loading from "@/components/Loading";
import LeaderboardTable from "@/components/LeaderboardTable";
import Schema from "@/types/Schema";
import getSchema from "@/services/getSchema";

export default function Groups() {
  const [activeTableIndex, setActiveTableIndex] = useState<number>(0);
  const [tables, setTables] = useState<GroupsTable[] | undefined>();
  const [schema, setSchema] = useState<Schema | undefined>();

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schemaPromise = getSchema(controller.signal);
      const groupsPromise = getGroupsTables(controller.signal);

      const schemaResult = await schemaPromise;
      setSchema(schemaResult);
      const groups = await groupsPromise;
      setTables(groups);
    }
    void fetchData();
    return () => controller.abort();
  }, []);

  if (tables === undefined || schema === undefined) {
    return <Loading />;
  }

  if (tables.length === 0) {
    return (
      <div>
        <PageTitle
          title="Results"
          subtitle="Groupings of the processes, methods, and metrics involved in evaluating models, particularly in the context of natural language understanding and question answering."
          className="mb-16"
        />
        <div>No groups found.</div>
      </div>
    );
  }

  return (
    <div>
      <PageTitle
        title="Results"
        subtitle="Groupings of the processes, methods, and metrics involved in evaluating models, particularly in the context of natural language understanding and question answering."
        className="mb-16"
      />
      <div>
        {tables.length > 1 ? (
          <Tabs>
            {tables.map((table, idx) => (
              <Tab
                key={idx}
                active={idx === activeTableIndex}
                onClick={() => setActiveTableIndex(idx)}
              >
                {table.title}
              </Tab>
            ))}
          </Tabs>
        ) : null}
        <LeaderboardTable
          key={`${activeTableIndex}`}
          schema={schema}
          groupTable={tables[activeTableIndex]}
          numRowsToDisplay={-1}
          sortColumnIndex={1}
          sortable={true}
        />
      </div>
    </div>
  );
}
