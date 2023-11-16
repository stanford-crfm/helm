import Instance from "@/types/Instance";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getInstancesByRunName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<Instance[]> {
  try {
    if (suite) {
      const instances = await fetch(
        getBenchmarkEndpoint(
          `/benchmark_output/runs/${suite}/${runName}/instances.json`,
        ),
        { signal },
      );

      return (await instances.json()) as Instance[];
    } else {
      const instances = await fetch(
        getBenchmarkEndpoint(
          `/benchmark_output/runs/${getBenchmarkSuite()}/${runName}/instances.json`,
        ),
        { signal },
      );

      return (await instances.json()) as Instance[];
    }
  } catch (error) {
    console.log(error);
    return [];
  }
}
