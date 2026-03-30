import MiniLeaderboard from "@/components/MiniLeaderboard";
import roborewardOverview from "@/assets/roboreward/overview.png";
import { Link } from "react-router-dom";

export default function RoboRewardLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        <strong>RoboReward</strong>: General-purpose Vision-Language Reward
        Models for Robotics
      </h1>

      <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-4 md:gap-8 my-8">
        <a
          className="px-10 btn rounded-md"
          href="https://arxiv.org/abs/2601.00675"
        >
          Paper
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://crfm.stanford.edu/helm/robo-reward-bench/latest/#/leaderboard"
        >
          Full Leaderboard
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://huggingface.co/datasets/teetone/RoboReward"
        >
          RoboReward Dataset
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://huggingface.co/teetone/RoboReward-4B"
        >
          RoboReward 4B
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://huggingface.co/teetone/RoboReward-8B"
        >
          RoboReward 8B
        </a>
      </div>

      <p className="my-4">
        A well-designed reward is critical for effective reinforcement
        learning-based policy improvement. In real-world robotic domains,
        obtaining such rewards typically requires either labor-intensive human
        labeling or brittle, handcrafted objectives. Vision-language models
        (VLMs) have shown promise as automatic reward models, yet their
        effectiveness on real robot tasks is poorly understood.
      </p>
      <p className="my-4">
        In this work, we aim to close this gap by introducing (1){" "}
        <strong>RoboReward</strong>, a robotics reward dataset and benchmark
        built on large-scale real-robot corpora from Open X-Embodiment (OXE) and
        RoboArena, and (2) vision-language reward models trained on this dataset
        (<strong>RoboReward 4B/8B</strong>). Because OXE is success-heavy and
        lacks failure examples, we propose a{" "}
        <em>negative examples data augmentation</em> pipeline that generates
        calibrated <em>negatives</em> and <em>near-misses</em> via
        counterfactual relabeling of successful episodes and temporal clipping
        to create partial-progress outcomes from the same videos.
      </p>
      <p className="my-4">
        Using this framework, we produce an extensive training and evaluation
        dataset that spans diverse tasks and embodiments and enables systematic
        evaluation of whether state-of-the-art VLMs can reliably provide rewards
        for robotics. Our evaluation of leading open-weight and proprietary VLMs
        reveals that no model excels across all tasks, underscoring substantial
        room for improvement. We then train general-purpose 4B- and 8B-parameter
        models that outperform much larger VLMs in assigning rewards for
        short-horizon robotic tasks. Finally, we deploy the 8B reward VLM in
        real-robot reinforcement learning and find that it improves policy
        learning over Gemini Robotics-ER 1.5, a frontier physical reasoning VLM
        trained on robotics data, by a large margin, while substantially
        narrowing the gap to RL training with human-provided rewards.
      </p>

      <div className="my-12 flex flex-col lg:flex-row gap-8 items-start">
        <div className="flex-[2]">
          <img
            src={roborewardOverview}
            alt="RoboReward overview"
            className="w-full max-w-full rounded-lg shadow"
          />
        </div>
        <div className="flex-[1]">
          <MiniLeaderboard />
          <Link
            to="leaderboard"
            className="px-4 mx-3 mt-3 btn bg-white rounded-md"
          >
            <span>See more</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
