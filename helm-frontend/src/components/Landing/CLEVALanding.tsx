import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

import cuhk from "@/assets/logos/cuhk.png";

export default function CLEVALanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        Chinese Language Models EVAluation Platform (CLEVA)
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <div className="text-center">
            <a href="https://www.cuhk.edu.hk/">
              <img src={cuhk} alt="Logo" className="inline h-12 mx-4 my-4" />
            </a>
          </div>
          <p>
            In collaboration with the{" "}
            <a
              href="https://lwwangcse.github.io/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              LaVi Lab
            </a>{" "}
            team from{" "}
            <a
              href="https://www.cuhk.edu.hk/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              The Chinese University of Hong Kong (CUHK)
            </a>
            , we introduce the{" "}
            <strong className="font-bold">
              Chinese Language Models EVAluation Platform (CLEVA)
            </strong>{" "}
            leaderboard on HELM. CLEVA is a comprehensive Chinese-language
            benchmark for holistic evaluation of Chinese-language LLMs, and
            employs a standardized workflow to assess LLMs' performance across
            various dimensions.
          </p>
          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://arxiv.org/abs/2308.04813"
            >
              Paper
            </a>
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
              Full Leaderboard
            </a>
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
