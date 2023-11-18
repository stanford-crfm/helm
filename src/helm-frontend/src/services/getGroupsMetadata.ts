import type GroupsMetadata from "@/types/GroupsMetadata";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getGroupsMetadata(
  signal: AbortSignal,
): Promise<GroupsMetadata> {
  try {
    const groups = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/groups_metadata.json`),
      { signal },
    );

    return (await groups.json()) as GroupsMetadata;
  } catch (error) {
    console.log(error);
    return {} as GroupsMetadata;
  }
}
