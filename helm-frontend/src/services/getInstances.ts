import Instance from "@/types/Instance";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getInstancesByRunName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<Instance[]> {
  try {
    const instances = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${runName}/instances.json`,
      ),
      { signal },
    );

    return (await instances.json()) as Instance[];
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
