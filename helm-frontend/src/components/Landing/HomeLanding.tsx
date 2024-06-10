import { useEffect, useState } from "react";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import CardGrid from "@/components/CardGrid";
import SimpleHero from "@/components/SimpleHero";

//import languageModelHelm from "@/assets/language-model-helm.png";
//import scenariosByMetrics from "@/assets/scenarios-by-metrics.png";
//import taxonomyScenarios from "@/assets/taxonomy-scenarios.png";
import ai21 from "@/assets/logos/ai21.png";
import alephAlpha from "@/assets/logos/aleph-alpha.png";
import anthropic from "@/assets/logos/anthropic.png";
import bigscience from "@/assets/logos/bigscience.png";
import cohere from "@/assets/logos/cohere.png";
import eleutherai from "@/assets/logos/eleutherai.png";
import google from "@/assets/logos/google.png";
import meta from "@/assets/logos/meta.png";
import microsoft from "@/assets/logos/microsoft.png";
import mistral from "@/assets/logos/mistral.png";
import nvidia from "@/assets/logos/nvidia.png";
import openai from "@/assets/logos/openai.png";
import tii from "@/assets/logos/tii.png";
import together from "@/assets/logos/together.png";
import tsinghuaKeg from "@/assets/logos/tsinghua-keg.png";
import writer from "@/assets/logos/writer.png";
import yandex from "@/assets/logos/yandex.png";
import zeroOne from "@/assets/logos/01.png";

const logos = [
  ai21,
  alephAlpha,
  anthropic,
  bigscience,
  cohere,
  eleutherai,
  google,
  meta,
  microsoft,
  mistral,
  nvidia,
  openai,
  tii,
  together,
  tsinghuaKeg,
  writer,
  yandex,
  zeroOne,
];

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
      <div className="mx-auto text-lg px-16">
        <div className="container mb-12 mx-auto text-lg px-16">
          <div className="flex flex-col sm:flex-row justify-center mt-10 mb-10 flex gap-2 sm:gap-8 md:gap-32">
            {" "}
            <h1 className="text-4xl  mx-4 mt-40">
              <strong>Our Partners</strong>
            </h1>
          </div>

          <ol className="my-8 flex flex-col gap-32">
            <li>
              <div className="flex flex-wrap justify-center max-w-[1100px] mx-auto w-auto">
                {logos.map((logo, idx) => (
                  <div className="w-24 h-24 flex items-center m-6" key={idx}>
                    <img
                      src={logo}
                      alt="Logo"
                      className="mx-auto block"
                      sizes="100vw"
                    />
                  </div>
                ))}
              </div>
            </li>
          </ol>
        </div>
      </div>
    </>
  );
}
