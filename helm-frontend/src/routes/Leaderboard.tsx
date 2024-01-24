import { useEffect, useState } from "react";
import PageTitle from "@/components/PageTitle";
import Tab from "@/components/Tab";
import Tabs from "@/components/Tabs";
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

export default function Leaderboard() {
  const defaultGroup = { title: "Core Scenarios", name: "core_scenarios" };
  const [allGroupData, setAllGroupData] = useState<GroupDisplayData[]>([]);
  const [selectedGroupDisplayData, setSelectedGroupDisplayData] =
    useState(defaultGroup);
  const [groupsTables, setGroupsTables] = useState<GroupsTable[]>([]);
  const [groupMetadata, setGroupMetadata] = useState<
    GroupMetadata | undefined
  >();
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [activeGroup, setActiveGroup] = useState<number>(0);

  function findMatchingGroup(
    allGroupData: GroupDisplayData[],
    target: string,
  ): GroupDisplayData {
    console.log(allGroupData, target);
    const searchResult = allGroupData.find((group) => group.title === target);
    if (searchResult != undefined) {
      return searchResult;
    } else {
      return defaultGroup;
    }
  }

  function updateLeaderboard(allGroupData: GroupDisplayData[], target: string) {
    setSelectedGroupDisplayData(findMatchingGroup(allGroupData, target));
  }

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
  }, [selectedGroupDisplayData]);

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
        <div className="flex flex-row justify-between">
          <PageTitle
            title={"HELM Leaderboard"}
            subtitle={
              "HELM is a framework for evaluating foundation models. Our leaderboard shows how the various models (with particular adaptation procedures) perform across different groups of scenarios and different metrics."
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
              value={selectedGroupDisplayData.title}
              onChange={(e) => updateLeaderboard(allGroupData, e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring focus:border-blue-300 rounded-md"
            >
              {allGroupData.map((group, index) => (
                <option key={index} value={group.title}>
                  {group.title}
                </option>
              ))}
            </select>
          </div>
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
        <LeaderboardTables
          groupsTables={groupsTables}
          activeGroup={activeGroup}
          ignoreHref={true}
        />
      </>
    </>
  );
}
