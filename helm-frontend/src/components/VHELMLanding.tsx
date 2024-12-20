import { Link } from "react-router-dom";

import MiniLeaderboard from "@/components/MiniLeaderboard";

import vhelmFrameworkImage from "@/assets/vhelm/vhelm-framework.png";
import vhelmModelImage from "@/assets/vhelm/vhelm-model.png";
import vhelmAspectsImage from "@/assets/vhelm/vhelm-aspects.png";

export default function VHELMLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        Holistic Evaluation of Vision-Language Models
      </h1>

      <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-8 md:gap-32 my-8">
        <a
          className="px-10 btn rounded-md"
          // TODO: update with VHELM paper link
          href="https://arxiv.org/abs/2410.07112"
        >
          Paper
        </a>
        <a className="px-10 btn rounded-md" href="#/leaderboard">
          Leaderboard
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://github.com/stanford-crfm/helm"
        >
          Github
        </a>
      </div>
      <p className="my-4">
        Current benchmarks for assessing vision-language models (VLMs) often
        focus on their perception or problem-solving capabilities and neglect
        other critical aspects such as fairness, multilinguality, or toxicity.
        Furthermore, they differ in their evaluation procedures and the scope of
        the evaluation, making it difficult to compare models. To address these
        issues, we extend the HELM framework to VLMs to present the Holistic
        Evaluation of Vision Language Models (VHELM). To address these issues,
        we introduce VHELM, built on HELM for language models. VHELM aggregates
        various datasets to cover one or more of the 9 aspects:{" "}
        <b>visual perception</b>, <b>bias</b>, <b>fairness</b>, <b>knowledge</b>
        , <b>multilinguality</b>, <b>reasoning</b>, <b>robustness</b>,{" "}
        <b>safety</b>, and <b>toxicity</b>. In doing so, we produce a
        comprehensive, multi-dimensional view of the capabilities of the VLMs
        across these important factors. In addition, we standardize the standard
        inference parameters, methods of prompting, and evaluation metrics to
        enable fair comparisons across models. Our framework is designed to be
        lightweight and automatic so that evaluation runs are cheap and fast.
        For transparency, we release the raw model generations and complete
        results on this website.
      </p>
      <p className="my-4 font-bold">
        VHELM is intended to be a living benchmark. We hope to continue adding
        new datasets, models and metrics over time, so please stay tuned!
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
          <MiniLeaderboard />
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
