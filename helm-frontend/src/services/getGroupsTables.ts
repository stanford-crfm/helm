import type GroupsTable from "@/types/GroupsTable";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export function getGroupsTablesJsonUrl(): string {
  return getBenchmarkEndpoint(`${getVersionBaseUrl()}/groups.json`);
}

export default async function getGroupsTables(
  signal: AbortSignal,
): Promise<GroupsTable[]> {
  try {
    const groups = await fetch(getGroupsTablesJsonUrl(), { signal });

    return (await groups.json()) as GroupsTable[];
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
