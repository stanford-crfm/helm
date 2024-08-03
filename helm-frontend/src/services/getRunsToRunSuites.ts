import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkRelease from "@/utils/getBenchmarkRelease";

export default async function getRunsToRunSuites(
  signal: AbortSignal,
): Promise<Record<string, string>> {
  try {
    const runsToRunSuites = await fetch(
      getBenchmarkEndpoint(
        `/releases/${getBenchmarkRelease()}/runs_to_run_suites.json`,
      ),
      { signal },
    );

    return (await runsToRunSuites.json()) as Record<string, string>;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return {};
  }
}
