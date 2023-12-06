import type Schema from "@/types/Schema";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export function getSchemaJsonUrl(): string {
  return getBenchmarkEndpoint(`${getVersionBaseUrl()}/schema.json`);
}

export default async function getSchema(signal: AbortSignal): Promise<Schema> {
  try {
    const resp = await fetch(getSchemaJsonUrl(), { signal });
    const data = await resp.text();
    const schema = JSON.parse(data) as Schema;
    return schema;
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
