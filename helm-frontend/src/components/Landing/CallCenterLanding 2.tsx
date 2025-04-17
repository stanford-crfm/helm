import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function CallCenterLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        HELM Call Center Leaderboard
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            LLMs show great potential for applications for the call center, yet
            there is a lack of domain-specific and ecologically-valid
            evaluations in this domain. To address this, we introduce the{" "}
            <strong className="font-bold">HELM Call Center leaderboard</strong>.
            The HELM Call Center leaderboard evaluates leading LLMs on a
            summarization task over a dataset of real helpdesk call transcripts
            provided by Accenture. The quality of the summaries is evaluated
            using LLM-as-judge with an ensemble of 3 models. We hope that this
            leaderboard provides some initial insights into the potential of
            LLMs in this domain.
          </p>
          <p className="my-4">
            This leaderboard was produced through research collaboration with{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://www.accenture.com/"
            >
              Accenture
            </a>
            , and was funded by the{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://hai.stanford.edu/corporate-affiliate-program"
            >
              HAI Corporate Affiliate Program
            </a>
            .
          </p>
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
