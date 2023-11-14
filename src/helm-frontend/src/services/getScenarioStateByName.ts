import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export function getScenarioStateByNameUrl(runName: string): string {
  return getBenchmarkEndpoint(
    `${getVersionBaseUrl()}/${runName}/scenario_state.json`,
  );
}
