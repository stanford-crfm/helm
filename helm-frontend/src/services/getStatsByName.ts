import type Stat from "@/types/Stat";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getStatsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<Stat[]> {
  try {
    const stats = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${runName}/stats.json`,
      ),
      { signal },
    );

    return (await stats.json()) as Stat[];
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
