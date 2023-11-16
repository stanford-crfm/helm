import type DisplayRequest from "@/types/DisplayRequest";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getDisplayRequestsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<DisplayRequest[]> {
  try {
    if (suite) {
      const displayRequest = await fetch(
        getBenchmarkEndpoint(
          `/benchmark_output/runs/${suite}/${runName}/display_requests.json`,
        ),
        { signal },
      );

      return (await displayRequest.json()) as DisplayRequest[];
    } else {
      const displayRequest = await fetch(
        getBenchmarkEndpoint(
          `/benchmark_output/runs/${getBenchmarkSuite()}/${runName}/display_requests.json`,
        ),
        { signal },
      );

      return (await displayRequest.json()) as DisplayRequest[];
    }
  } catch (error) {
    console.log(error);
    return [];
  }
}
