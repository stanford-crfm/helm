import { useEffect, useState } from "react";
import PageTitle from "@/components/PageTitle";
import LeaderboardTables from "@/components/LeaderboardTables";
import type GroupsTable from "@/types/GroupsTable";
import type GroupMetadata from "@/types/GroupMetadata";
import getGroupsTablesByName from "@/services/getGroupTablesByName";
import getGroupsMetadata from "@/services/getGroupsMetadata";
import Loading from "@/components/Loading";
import getGroupsTables from "@/services/getGroupsTables";

interface GroupDisplayData {
  title: string;
  name: string;
}

export default function MiniLeaderboard() {
  const defaultGroup = { title: "Core Scenarios", name: "core_scenarios" };
  const selectedGroupDisplayData = defaultGroup;
  const [allGroupData, setAllGroupData] = useState<GroupDisplayData[]>([]);
  const [groupsTables, setGroupsTables] = useState<GroupsTable[]>([]);
  const [groupMetadata, setGroupMetadata] = useState<
    GroupMetadata | undefined
  >();
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const activeGroup = 0;
  console.log(allGroupData);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      if (selectedGroupDisplayData.name === undefined) {
        return;
      }
      const groups = await getGroupsTables(controller.signal);
      const result: GroupDisplayData[] = [];
      groups.forEach((group) => {
        group.rows.forEach((row) => {
          result.push({
            title: String(row[0].value),
            name: row[0].href.replace("?group=", ""),
          });
        });
      });
      setAllGroupData(result);

      const [group, metadata] = await Promise.all([
        getGroupsTablesByName(selectedGroupDisplayData.name, controller.signal),
        getGroupsMetadata(controller.signal),
      ]);
      setGroupsTables(group);
      setGroupMetadata(metadata[selectedGroupDisplayData.name]);
      setIsLoading(false);
    }

    void fetchData();
    return () => controller.abort();
  }, [selectedGroupDisplayData.name]);

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
      <>
        <LeaderboardTables
          groupsTables={groupsTables}
          activeGroup={activeGroup}
          ignoreHref={true}
          filtered
          numModelsToAutoFilter={6}
          filteredCols={[0, 1]}
        />
      </>
    </>
  );
}
