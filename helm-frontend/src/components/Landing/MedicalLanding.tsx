import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function MedicalLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">HELM Medical</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-2">
            With the increasing scale and impact of language models, there has
            also been interest interest in using language models in the medical
            domain. However, the capabilities and risks of these models are not
            well-understood, and there is significant potential for harm in the
            medical setting.
          </p>
          <p className="my-2">
            To address this, we present the{" "}
            <a className="font-bold" href="https://arxiv.org/abs/2405.09605">
              HELM Medical
            </a>{" "}
            leaderboard for evaluation of language models in the medical domain.
            The HELM Medical leaderboard presents evaluations of leading
            general-purpose language models as well as language models
            fine-tuned on the medical domain. These models are evaluated on a
            range of medical tasks based on the benchmarks used in{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://arxiv.org/abs/2212.13138"
            >
              Singhal et al. 2022
            </a>
            . We hope that this leaderboard encourages further work in
            evaluating language models on tasks from the medical domain.
          </p>
          <div className="flex flex-row justify-center my-4">
            <Link to="leaderboard" className="px-10 btn rounded-md mx-4">
              Full Leaderboard
            </Link>
          </div>
        </div>
        <div
          className="py-2 pb-6 rounded-3xl bg-gray-100 h-full" // Stretched to full height
          style={{ maxWidth: "100%" }}
        >
          <MiniLeaderboard />
          <div className="flex justify-end">
            <Link to="leaderboard">
              <button className="px-4 mx-3 mt-1 btn bg-white rounded-md">
                <span>See More</span>
              </button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
