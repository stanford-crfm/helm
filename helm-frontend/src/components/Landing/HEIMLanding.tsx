import { useEffect, useState } from "react";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import heimLogo from "@/assets/heim-logo.png";
import MetricsList from "@/components/MetricsList";
import ModelsList from "@/components/ModelsList";
import ScenariosList from "@/components/ScenariosList";
import type MetricFieldMap from "@/types/MetricFieldMap";

export default function HEIMLanding() {
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

  const metricFieldMap: MetricFieldMap | undefined = schema
    ? schema.metrics.reduce((acc, cur) => {
        acc[cur.name] = cur;
        return acc;
      }, {} as MetricFieldMap)
    : undefined;
  return (
    <div className="container mx-auto px-16 text-base">
      <div className="container max-w-screen-lg mx-auto">
        <img className="mx-auto w-96" src={heimLogo} alt="HEIM Logo" />
      </div>
      <h1 className="text-3xl my-8 font-bold text-center">
        Holistic Evaluation of Text-To-Image Models
      </h1>
      <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-8 md:gap-32 my-8">
        <a
          className="px-10 btn rounded-md"
          href="https://arxiv.org/abs/2311.04287"
        >
          Paper
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://github.com/stanford-crfm/helm"
        >
          Github
        </a>
      </div>
      <p className="my-2">
        Significant effort has recently been made in developing text-to-image
        generation models, which take textual prompts as input and generate
        images. As these models are widely used in real-world applications,
        there is an urgent need to comprehensively understand their capabilities
        and risks. However, existing evaluations primarily focus on image-text
        alignment and image quality. To address this limitation, we introduce a
        new benchmark,{" "}
        <strong>Holistic Evaluation of Text-To-Image Models (HEIM)</strong>.
      </p>
      <p className="my-2">
        We identify 12 different aspects that are important in real-world model
        deployment, including:
      </p>
      <ul className="my-2 list-disc list-inside unreset">
        <li>image-text alignment</li>
        <li>image quality</li>
        <li>aesthetics</li>
        <li>originality</li>
        <li>reasoning</li>
        <li>knowledge</li>
        <li>bias</li>
        <li>toxicity</li>
        <li>fairness</li>
        <li>robustness</li>
        <li>multilinguality</li>
        <li>efficiency</li>
      </ul>
      <p className="my-2">
        By curating scenarios encompassing these aspects, we evaluate
        state-of-the-art text-to-image models using this benchmark. Unlike
        previous evaluations that focused on alignment and quality, HEIM
        significantly improves coverage by evaluating all models across all
        aspects. Our results reveal that no single model excels in all aspects,
        with different models demonstrating strengths in different aspects.
      </p>
      <p className="my-2">
        For full transparency, this website contains all the prompts, generated
        images and the results for the automated and human evaluation metrics.
      </p>
      <p className="my-2">
        Inspired by HELM, we decompose the model evaluation into four key
        components: aspect, scenario, adaptation, and metric:
      </p>
      <div className="container max-w-screen-lg mx-auto my-8">
        <img
          src="https://crfm.stanford.edu/heim/latest/images/heim-main.png"
          alt="HEIM scenarios, prompts, images and metrics"
        />
      </div>

      {schema && metricFieldMap ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <ModelsList models={schema.models} />
          <ScenariosList runGroups={schema.run_groups} />
          <MetricsList
            metricFieldMap={metricFieldMap}
            metricGroups={schema.metric_groups}
          />
        </div>
      ) : null}
    </div>
  );
}
