import { parse } from "yaml";

import type Schema from "@/types/Schema";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import getBenchmarkRelease from "@/utils/getBenchmarkRelease";

export default async function getSchema(signal: AbortSignal): Promise<Schema> {
  try {
    if (getBenchmarkRelease()) {
      const resp = await fetch(
        `https://crfm.stanford.edu/helm/${getBenchmarkRelease()}/schema.yaml`,
        { signal },
      );
      const data = await resp.text();
      const schema = parse(data) as Schema;

      return schema;
    } else {
      const resp = await fetch(
        `https://crfm.stanford.edu/helm/${getBenchmarkSuite()}/schema.yaml`,
        { signal },
      );
      const data = await resp.text();
      const schema = parse(data) as Schema;

      return schema;
    }
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
