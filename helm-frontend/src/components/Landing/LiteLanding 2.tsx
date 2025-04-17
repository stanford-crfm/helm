import { useEffect, useState } from "react";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import ModelsList from "@/components/ModelsList";
import ScenariosList from "@/components/ScenariosList";
import Hero from "@/components/Hero";

//import languageModelHelm from "@/assets/language-model-helm.png";
//import scenariosByMetrics from "@/assets/scenarios-by-metrics.png";
//import taxonomyScenarios from "@/assets/taxonomy-scenarios.png";

export default function LegacyLanding() {
  const [schema, setSchema] = useState<Schema | undefined>(undefined);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schema = await getSchema(controller.signal);
      setSchema(schema);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  if (!schema) {
    return null;
  }

  return (
    <>
      <Hero />
      <div className="mx-auto text-lg px-16">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
            <ModelsList models={schema.models} />
            <ScenariosList runGroups={schema.run_groups} />
          </div>
        </div>
      </div>
    </>
  );
}
