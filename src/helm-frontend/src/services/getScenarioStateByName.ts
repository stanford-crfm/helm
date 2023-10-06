import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export function getScenarioStateByNameUrl(runName: string): string {
  return getBenchmarkEndpoint(
    `/benchmark_output/runs/${getBenchmarkSuite()}/${runName}/scenario_state.json`,
  );
}
