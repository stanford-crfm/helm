import type Stat from "@/types/Stat";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getStatsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<Stat[]> {
  try {
    if (suite) {
      const stats = await fetch(
        getBenchmarkEndpoint(
          `/benchmark_output/runs/${suite}/${runName}/stats.json`,
        ),
        { signal },
      );

      return (await stats.json()) as Stat[];
    } else {
      const stats = await fetch(
        getBenchmarkEndpoint(
          `/benchmark_output/runs/${getBenchmarkSuite()}/${runName}/stats.json`,
        ),
        { signal },
      );

      return (await stats.json()) as Stat[];
    }
  } catch (error) {
    console.log(error);
    return [];
  }
}
