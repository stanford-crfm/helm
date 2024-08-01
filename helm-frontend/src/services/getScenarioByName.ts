import Scenario from "@/types/Scenario";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getScenarioByName(
  scenarioName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<Scenario | undefined> {
  try {
    const scenario = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${scenarioName}/scenario.json`,
      ),
      { signal },
    );

    return (await scenario.json()) as Scenario;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return undefined;
  }
}
