import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";
import ReleaseSummary from "@/types/ReleaseSummary";

export default async function getReleaseSummary(
  signal: AbortSignal,
): Promise<ReleaseSummary> {
  try {
    const summary = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/summary.json`),
      { signal },
    );

    return (await summary.json()) as ReleaseSummary;
  } catch (error) {
    console.log(error);
    return {
      release: undefined,
      suites: undefined,
      suite: undefined,
      date: "",
    };
  }
}
