import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function FinanceLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">HELM Finance</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            LLMs show great potential for applications in the financial domain,
            yet there is a lack of financial-domain evaluations for LLMs. To
            address this, we introduce the{" "}
            <strong className="font-bold">HELM Finance</strong> leaderboard. The
            HELM Finance leaderboard evaluates leading LLMs on a three financial
            benchmarks (FinQA, FinanceBench, BANKING77) that utilize real
            financial documents. Like all other HELM leaderboards, the HELM
            Finance leaderboard provides full prompt-level transparency, and the
            results can be fully reproduced using the open-source HELM
            framework. We hope that this leaderboard provides valuable insights
            for financial practitioners.
          </p>

          <p className="my-4">
            This leaderboard was produced through a collaboration with{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://www.wellsfargo.com/"
            >
              Wells Fargo
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
