import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export function getScenarioStateByNameUrl(
  runName: string,
  suite?: string,
): string {
  return getBenchmarkEndpoint(
    `/runs/${suite || getBenchmarkSuite()}/${runName}/scenario_state.json`,
  );
}
