import type Schema from "@/types/Schema";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getSchema(signal: AbortSignal): Promise<Schema> {
  try {
    const resp = await fetch(
      getBenchmarkEndpoint(`${getVersionBaseUrl()}/schema.json`),
      { signal },
    );

    return (await resp.json()) as Schema;
  } catch (error) {
    console.log(error);
    return {
      adapter: [],
      metric_groups: [],
      metrics: [],
      models: [],
      perturbations: [],
      run_groups: [],
    } as Schema;
  }
}
