import RunSpec from "@/types/RunSpec";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getRunSpecs(
  signal: AbortSignal,
): Promise<RunSpec[]> {
  try {
    const runSpecs = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/run_specs.json`),
      { signal },
    );

    return (await runSpecs.json()) as RunSpec[];
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
