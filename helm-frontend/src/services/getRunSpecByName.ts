import type RunSpec from "@/types/RunSpec";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export function getRunSpecByNameUrl(runName: string, suite?: string): string {
  return getBenchmarkEndpoint(
    `/runs/${suite || getBenchmarkSuite()}/${runName}/run_spec.json`,
  );
}
export default async function getRunSpecByName(
  runName: string,
  signal: AbortSignal,
): Promise<RunSpec | undefined> {
  try {
    const runSpec = await fetch(getRunSpecByNameUrl(runName), { signal });

    return (await runSpec.json()) as RunSpec;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return undefined;
  }
}
