import type GroupsTable from "@/types/GroupsTable";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export function getGroupsTablesJsonUrl(): string {
  return getBenchmarkEndpoint(
    `/benchmark_output/runs/${getBenchmarkSuite()}/groups.json`,
  );
}

export default async function getGroupsTables(
  signal: AbortSignal,
): Promise<GroupsTable[]> {
  try {
    const groups = await fetch(getGroupsTablesJsonUrl(), { signal });

    return (await groups.json()) as GroupsTable[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
