import { useEffect, useState } from "react";
import { ArrowDownTrayIcon } from "@heroicons/react/24/solid";
import type GroupsTable from "@/types/GroupsTable";
import getGroupsTables, {
  getGroupsTablesJsonUrl,
} from "@/services/getGroupsTables";
import PageTitle from "@/components/PageTitle";
import Tabs from "@/components/Tabs";
import Tab from "@/components/Tab";
import GroupsTables from "@/components/GroupsTables";
import Loading from "@/components/Loading";

export default function Groups() {
  const [activeGroup, setActiveGroup] = useState<number>(0);
  const [tabs, setTabs] = useState<string[]>([]);
  const [groupsTables, setGroupsTables] = useState<GroupsTable[]>(
    [] as GroupsTable[],
  );

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const groups = await getGroupsTables(controller.signal);
      setGroupsTables(groups);
      setTabs(groups.map((group) => group.title));
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  if (groupsTables.length === 0) {
    return <Loading />;
  }

  return (
    <>
      <div className="flex justify-between">
        <PageTitle
          title="Results"
          subtitle="Groupings of the processes, methods, and metrics involved in evaluating models, particularly in the context of natural language understanding and question answering."
          className="mb-16"
        />

        <a
          className="flex link-primary space-between items-center self-end link link-hover block"
          href={getGroupsTablesJsonUrl()}
          download="true"
          target="_blank"
        >
          <ArrowDownTrayIcon className="w-6 h-6 mr-2" /> JSON
        </a>
      </div>
      <div>
        <Tabs>
          {tabs.map((tab, idx) => (
            <Tab
              onClick={() => setActiveGroup(idx)}
              key={idx}
              active={activeGroup === idx}
              size="lg"
            >
              {tab}
            </Tab>
          ))}
        </Tabs>
      </div>
      <div className="mt-8">
        <GroupsTables
          sortable={false}
          groupsTables={groupsTables}
          activeGroup={activeGroup}
        />
      </div>
    </>
  );
}
