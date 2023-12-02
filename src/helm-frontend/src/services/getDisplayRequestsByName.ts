import type DisplayRequest from "@/types/DisplayRequest";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getDisplayRequestsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<DisplayRequest[]> {
  try {
    const displayRequest = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/runs/${
          suite || getBenchmarkSuite()
        }/${runName}/display_requests.json`,
      ),
      { signal },
    );

    return (await displayRequest.json()) as DisplayRequest[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
