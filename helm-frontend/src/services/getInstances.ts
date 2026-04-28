import Instance from "@/types/Instance";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import isScenarioEncrypted from "@/utils/isScenarioEncrypted";


export default async function getInstancesByRunName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
  userAgreed?: boolean,
): Promise<Instance[]> {
  try {
    const response = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${runName}/instances.json`,
      ),
      { signal },
    );
    const instances = (await response.json()) as Instance[];

    if (isScenarioEncrypted(runName) && !userAgreed) {
      return instances;
    }

    return instances;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
