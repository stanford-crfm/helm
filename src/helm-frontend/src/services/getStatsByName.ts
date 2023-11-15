import type Stat from "@/types/Stat";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getStatsByName(
  runName: string,
  signal: AbortSignal,
): Promise<Stat[]> {
  try {
    const stats = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/${runName}/stats.json`),
      { signal },
    );

    return (await stats.json()) as Stat[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
