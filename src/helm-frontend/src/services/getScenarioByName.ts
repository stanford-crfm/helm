import Scenario from "@/types/Scenario";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getScenarioByName(
  scenarioName: string,
  signal: AbortSignal,
): Promise<Scenario | undefined> {
  try {
    const scenario = await fetch(
      getBenchmarkEndpoint(
        `${getVersionBaseUrl()}/${scenarioName}/scenario.json`,
      ),
      { signal },
    );

    return (await scenario.json()) as Scenario;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}
