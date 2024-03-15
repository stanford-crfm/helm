import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getRunsToRunSuites(
  signal: AbortSignal,
): Promise<Record<string, string>> {
  try {
    const runsToRunSuites = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/runs_to_run_suites.json`),
      { signal },
    );

    return (await runsToRunSuites.json()) as Record<string, string>;
  } catch (error) {
    console.log(error);
    return {};
  }
}
