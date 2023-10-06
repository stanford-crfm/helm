import { parse } from "yaml";

import type Schema from "@/types/Schema";

export default async function getSchema(signal: AbortSignal): Promise<Schema> {
  try {
    const resp = await fetch(
      "https://crfm.stanford.edu/helm/latest/schema.yaml",
      { signal },
    );
    const data = await resp.text();
    const schema = parse(data) as Schema;

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
