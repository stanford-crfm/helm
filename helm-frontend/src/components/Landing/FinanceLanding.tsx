import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import wellsfargo from "@/assets/logos/wellsfargo.png";

export default function FinanceLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">HELM Finance</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <div>
            <a href="https://wellsfargo.com/">
              <img
                src={wellsfargo}
                alt="Logo"
                className="mx-auto block my-4 w-48"
              />
            </a>
          </div>
          <p>
            In collaboration with{" "}
            <a
              href="https://www.wellsfargo.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              Wells Fargo
            </a>
            , we introduce the <span className="font-bold">HELM Finance</span>{" "}
            leaderboard for ecologically-valid evaluations of leading language
            models in the financial domain. The leaderboard evaluates the
            ability of language models to perform tasks from financial
            professions on publicly financial documents across a range of
            scenarios.
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
