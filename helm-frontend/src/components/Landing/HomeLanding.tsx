import { useEffect, useState } from "react";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import CardGrid from "@/components/CardGrid";
import SimpleHero from "@/components/SimpleHero";

//import languageModelHelm from "@/assets/language-model-helm.png";
//import scenariosByMetrics from "@/assets/scenarios-by-metrics.png";
//import taxonomyScenarios from "@/assets/taxonomy-scenarios.png";

export default function HomeLanding() {
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
      <SimpleHero />
      <div className="container py-5 mx-auto text-lg">
        <div className="flex flex-col sm:flex-row justify-center mb-10 flex sm:gap-8 md:gap-32">
          <h1 className="text-4xl mx-4 ">
            <strong>HELM Leaderboards</strong>
          </h1>
        </div>
      </div>
      <CardGrid />
    </>
  );
}
