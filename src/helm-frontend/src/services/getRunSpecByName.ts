import type RunSpec from "@/types/RunSpec";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export function getRunSpecByNameUrl(runName: string): string {
  return getBenchmarkEndpoint(
    `${getVersionBaseUrl()}/${runName}/run_spec.json`,
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
    console.log(error);
    return undefined;
  }
}
