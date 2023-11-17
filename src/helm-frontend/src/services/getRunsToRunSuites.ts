import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkRelease from "@/utils/getBenchmarkRelease";

export default async function getRunsToRunSuites(
  signal: AbortSignal,
): Promise<Record<string, string>> {
  try {
    const runsToRunSuites = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/releases/${getBenchmarkRelease()}/runs_to_run_suites.json`,
      ),
      { signal },
    );

    return (await runsToRunSuites.json()) as Record<string, string>;
  } catch (error) {
    console.log(error);
    return {};
  }
}
