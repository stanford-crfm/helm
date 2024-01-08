import { useEffect, useState } from "react";
import { useParams } from "react-router";
import PageTitle from "@/components/PageTitle";
import Tab from "@/components/Tab";
import Tabs from "@/components/Tabs";
//import GroupsCharts from "@/components/GroupsCharts";
import GroupsTables from "@/components/GroupsTables";
import type GroupsTable from "@/types/GroupsTable";
import type GroupMetadata from "@/types/GroupMetadata";
import getGroupsTablesByName from "@/services/getGroupTablesByName";
import getGroupsMetadata from "@/services/getGroupsMetadata";
import Loading from "@/components/Loading";

export default function Group() {
  const { groupName } = useParams();
  const [groupsTables, setGroupsTables] = useState<GroupsTable[]>([]);
  const [groupMetadata, setGroupMetadata] = useState<
    GroupMetadata | undefined
  >();
  const [isLoading, setIsLoading] = useState<boolean>(true);
  //const [showChart, setShowChart] = useState<boolean>(false);
  const [activeGroup, setActiveGroup] = useState<number>(0);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      if (groupName === undefined) {
        return;
      }

      const [group, metadata] = await Promise.all([
        getGroupsTablesByName(groupName, controller.signal),
        getGroupsMetadata(controller.signal),
      ]);
      setGroupsTables(group);
      setGroupMetadata(metadata[groupName]);
      setIsLoading(false);
    }

    void fetchData();
    return () => controller.abort();
  }, [groupName]);

  if (isLoading || groupMetadata === undefined) {
    return <Loading />;
  }

  if (groupsTables.length === 0) {
    return (
      <>
        <PageTitle
          title={groupMetadata.display_name}
          subtitle={groupMetadata.description}
          markdown={true}
          className="mr-8"
        />
        <div className="divider"></div>
        <p className="text-center mt-8">Group currently has no results.</p>
      </>
    );
  }

  return (
    <>
      <div className="flex flex-row justify-between">
        <PageTitle
          title={groupMetadata.display_name}
          subtitle={groupMetadata.description}
          markdown={true}
          className="mr-8 mb-16"
        />
        {/* 
        <button
          className="btn btn-xs btn-ghost self-end"
          onClick={() => setShowChart(!showChart)}
        >
          {showChart ? "Show Table" : "Show Chart"}
        </button>
        */}
      </div>
      <div className="overflow-x-auto">
        {groupsTables.length > 1 ? (
          <Tabs>
            {groupsTables.map((groupsTable, idx) => (
              <Tab
                key={idx}
                active={idx === activeGroup}
                onClick={() => setActiveGroup(idx)}
              >
                {groupsTable.title}
              </Tab>
            ))}
          </Tabs>
        ) : null}
      </div>

      {/*showChart ? (
        <GroupsCharts groupsTables={groupsTables} activeGroup={activeGroup} />
      ) : ( */}
      <GroupsTables
        groupsTables={groupsTables}
        activeGroup={activeGroup}
        ignoreHref={true}
      />
      {/* )*/}
    </>
  );
}
