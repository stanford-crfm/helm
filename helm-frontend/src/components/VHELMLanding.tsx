import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";

import MiniLeaderboard from "@/components/MiniLeaderboard";

import vhelmFrameworkImage from "@/assets/vhelm/vhelm-framework.png";
import vhelmModelImage from "@/assets/vhelm/vhelm-model.png";
import vhelmAspectsImage from "@/assets/vhelm/vhelm-aspects.png";

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
              Holistic Evaluation of Vision-Language Models
          </h1>

          <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-8 md:gap-32 my-8">
              <a
                  className="px-10 btn rounded-md"
                  // TODO: update with VHELM paper link
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
          <p className="my-4">
              To better understand VLMs, we introduce {" "}
              <em>Holistic Evaluation of Vision-Language Models (VHELM)</em> by
              extending the <a href="https://arxiv.org/abs/2211.09110">HELM</a>{" "}
              framework with the necessary adaptation methods to assess the
              performance of many prominent VLMs on across 9 aspects (see below).
          </p>
          <p className="my-4 font-bold">
              We will continue to incorporate new scenarios, models and metrics to the VHELM leaderboard
              to achieve holistic evaluation for vision-language models, so please stay tuned!
          </p>

          <div className="my-16 flex flex-col lg:flex-row items-center gap-8">
              <div className="flex-1 text-xl">
                  <img
                      src={vhelmModelImage}
                      alt="A vision-lanuage model (VLM) takes in an image and a text prompt and generates text."
                      className=""
                  />
                  <img
                      src={vhelmFrameworkImage}
                      alt="An example of an evaluation for an Aspect (Knowledge) - a Scenario (MMMU) undergoes Adaptation (multimodal multiple choice) for a Model (GPT-4 Omni), then Metrics (Exact match) are computed"
                      className=""
                  />
              </div>
              <div className="flex-1">
                  <MiniLeaderboard/>
                  <Link
                      to="leaderboard"
                      className="px-4 mx-3 mt-1 btn bg-white rounded-md"
                  >
                      <span>See More</span>
                  </Link>
              </div>
          </div>
          <div className="container max-w-screen-lg mx-auto my-8">
              <img
                  src={vhelmAspectsImage}
                  alt="An example of each aspect in VHELM: Visual Perception, Bias, Fairness, Knowledge, Multilinguality, Reasoning, Robustness, Toxicity Mitigation and Safety. "
                  className=""
              />
          </div>
      </div>
  );
}
