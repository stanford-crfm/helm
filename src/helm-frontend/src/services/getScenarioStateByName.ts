import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export function getScenarioStateByNameUrl(
  runName: string,
  suite?: string,
): string {
  if (suite) {
    return getBenchmarkEndpoint(
      `/benchmark_output/runs/${suite}/${runName}/scenario_state.json`,
    );
  } else {
    return getBenchmarkEndpoint(
      `/benchmark_output/runs/${getBenchmarkSuite()}/${runName}/scenario_state.json`,
    );
  }
}
