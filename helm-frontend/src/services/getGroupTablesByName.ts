import type GroupsTable from "@/types/GroupsTable";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getGroupsTablesByName(
  groupName: string,
  signal: AbortSignal,
): Promise<GroupsTable[]> {
  try {
    const group = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/groups/${groupName}.json`),
      { signal },
    );

    return (await group.json()) as GroupsTable[];
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
