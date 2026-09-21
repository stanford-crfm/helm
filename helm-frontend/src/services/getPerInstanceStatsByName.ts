import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export function getPerInstanceStatsByNameUrl(
  runName: string,
  suite?: string,
): string {
  return getBenchmarkEndpoint(
    `/runs/${suite || getBenchmarkSuite()}/${runName}/per_instance_stats.json`,
  );
}
