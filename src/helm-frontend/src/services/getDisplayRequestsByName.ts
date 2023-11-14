import type DisplayRequest from "@/types/DisplayRequest";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getDisplayRequestssByName(
  runName: string,
  signal: AbortSignal,
): Promise<DisplayRequest[]> {
  try {
    const displayRequest = await fetch(
      getBenchmarkEndpoint(
        `${getVersionBaseUrl()}/${runName}/display_requests.json`,
      ),
      { signal },
    );

    return (await displayRequest.json()) as DisplayRequest[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
