import type GroupsTable from "@/types/GroupsTable";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getGroupsTablesByName(
  groupName: string,
  signal: AbortSignal,
): Promise<GroupsTable[]> {
  try {
    const group = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/runs/${getBenchmarkSuite()}/groups/${groupName}.json`,
      ),
      { signal },
    );

    return await group.json() as GroupsTable[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
