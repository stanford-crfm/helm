import type GroupsMetadata from "@/types/GroupsMetadata";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getGroupsMetadata(
  signal: AbortSignal,
): Promise<GroupsMetadata> {
  try {
    const groups = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/runs/${getBenchmarkSuite()}/groups_metadata.json`,
      ),
      { signal },
    );

    return await groups.json() as GroupsMetadata;
  } catch (error) {
    console.log(error);
    return {} as GroupsMetadata;
  }
}
