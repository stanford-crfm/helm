import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import audioTable from "@/assets/audio/audio-table.png";

export default function AudioLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        Holistic Evaluation of Audio-Language Models
      </h1>

      <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-8 md:gap-32 my-8">
        <a
          className="px-10 btn rounded-md"
          href="https://arxiv.org/abs/2508.21376"
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
        Evaluations of{" "}
        <strong className="font-bold">audio-language models (ALMs)</strong>{" "}
        &mdash; multimodal models that take interleaved audio and text as input
        and output text &mdash; are hindered by the lack of standardized
        benchmarks; most benchmarks measure only one or two capabilities and
        omit evaluative aspects such as fairness or safety. Furthermore,
        comparison across models is difficult as separate evaluations test a
        limited number of models and use different prompting methods and
        inference parameters.
      </p>
      <p className="my-4">
        To address these shortfalls, we introduce{" "}
        <strong className="font-bold">AHELM</strong>, a benchmark that
        aggregates various datasets &mdash; including{" "}
        <strong className="font-bold">
          2 new synthetic audio-text datasets
        </strong>{" "}
        called <strong className="font-bold">PARADE</strong>, which evaluates
        the ALMs on avoiding stereotypes, and{" "}
        <strong className="font-bold">CoRe-Bench</strong>, which measures
        reasoning over conversational audio through inferential multi-turn
        question answering &mdash; to holistically measure the performance of
        ALMs across 10 aspects we have identified as important to the
        development and usage of ALMs:{" "}
        <em className="italic">audio perception</em>,{" "}
        <em className="italic">knowledge</em>,{" "}
        <em className="italic">reasoning</em>,{" "}
        <em className="italic">emotion detection</em>,{" "}
        <em className="italic">bias</em>, <em className="italic">fairness</em>,{" "}
        <em className="italic">multilinguality</em>,{" "}
        <em className="italic">robustness</em>,{" "}
        <em className="italic">toxicity</em>, and{" "}
        <em className="italic">safety</em>. We standardize the prompts,
        inference parameters, and evaluation metrics to ensure equitable
        comparisons across models.
      </p>
      <div className="my-16 flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-xl">
          <img
            src={audioTable}
            alt="An example of each aspect in AHELM: Auditory Perception, Knowledge, Reasoning, Emotion Detection, Bias, Fairness, Multilinguality, Robustness, Toxicity and Safety. "
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
    </div>
  );
}
