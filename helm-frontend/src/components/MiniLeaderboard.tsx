import { useEffect, useState } from "react";
import PageTitle from "@/components/PageTitle";
import MiniLeaderboardTables from "@/components/MiniLeaderboardTables";
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

interface Props {
  numModelsToAutoFilter?: number;
}
export default function MiniLeaderboard({ numModelsToAutoFilter = 6 }: Props) {
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
      if (result.length === 0) {
        throw new Error("Could not find any groups!");
      }
      const selectedGroupName = result[0].name;
      const [group, metadata] = await Promise.all([
        getGroupsTablesByName(selectedGroupName, controller.signal),
        getGroupsMetadata(controller.signal),
      ]);
      setGroupsTables(group);
      setGroupMetadata(metadata[selectedGroupName]);
      setIsLoading(false);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

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
        />
        <div className="divider"></div>
        <p className="text-center mt-8">Group currently has no results.</p>
      </>
    );
  }

  return (
    <>
      <>
        <MiniLeaderboardTables
          groupsTables={groupsTables}
          activeGroup={activeGroup}
          numModelsToAutoFilter={numModelsToAutoFilter}
          filteredCols={[0, 1]}
        />
      </>
    </>
  );
}
