import RunSpec from "@/types/RunSpec";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getRunSpecs(
  signal: AbortSignal,
): Promise<RunSpec[]> {
  try {
    const runSpecs = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/runs/${getBenchmarkSuite()}/run_specs.json`,
      ),
      { signal },
    );

    return await runSpecs.json() as RunSpec[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
