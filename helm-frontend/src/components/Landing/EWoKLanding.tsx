import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function EWoKLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        Elements of World Knowledge (EWoK)
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p>
            In collaboration with{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://www.language-intelligence-thought.net/"
            >
              Language, Intelligence & Thought (LIT) Lab
            </a>{" "}
            at{" "}
            <a
              href="https://www.gatech.edu/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              Georgia Tech
            </a>
            , we present the EWoK leaderboard. EWoK is a benchmark for
            evaluating world modeling in language models by testing their
            ability to use knowledge of a concept to match a target text with a
            plausible/implausible context. EWOK targets specific concepts from
            multiple knowledge domains known to be vital for world modeling in
            humans, including social interactions and spatial relations.
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
