import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";

import ModelsList from "@/components/ModelsList";
import MiniLeaderboard from "@/components/MiniLeaderboard";
import ScenariosList from "@/components/ScenariosList";

import vhelmFrameworkImage from "@/assets/vhelm/vhelm-framework.png";
import vhelmModelImage from "@/assets/vhelm/vhelm-model.png";

export default function VHELMLanding() {
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

  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        The First Steps to Holistic Evaluation of Vision-Language Models
      </h1>
      <p className="my-4">
        To better understand VLMs, we introduce the first version of{" "}
        <em>Holistic Evaluation of Vision-Language Models (VHELM)</em> by
        extending the <a href="https://arxiv.org/abs/2211.09110">HELM</a>{" "}
        framework with the necessary adaptation methods to assess the
        performance of 6 prominent VLMs on 3 standard VLM benchmarks.
      </p>
      <p className="my-4 font-bold">
        This is ongoing work to achieve holistic evaluation for vision-language
        models, so please stay tuned!
      </p>

      <div className="my-16 flex flex-col lg:flex-row items-center gap-8">
        <div className="flex-1 text-xl">
          <img
            src={vhelmFrameworkImage}
            alt="An image of a helm and the text 'This helm is a' is sent to a Vision-Language Model, which produces the text 'wheel for steering a ship...'"
            className=""
          />
          <img
            src={vhelmModelImage}
            alt="An example of an evaluation for an Aspect (Knowledge) - a Scenario (MMMU) undergoes Adaptation (multimodal multiple choice) for a Model (GPT-4 Vision), then Metrics (Exact match) are computed"
            className=""
          />
        </div>
        <div className="flex-1">
          <MiniLeaderboard numModelsToAutoFilter={10} />
          <Link
            to="leaderboard"
            className="px-4 mx-3 mt-1 btn bg-white rounded-md"
          >
            <span>See More</span>
          </Link>
        </div>
      </div>
      {schema === undefined ? null : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
          <ModelsList models={schema.models} />
          <ScenariosList runGroups={schema.run_groups} />
        </div>
      )}
    </div>
  );
}
