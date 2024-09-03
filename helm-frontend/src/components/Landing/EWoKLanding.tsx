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
            We present the{" "}
            <a href="https://arxiv.org/abs/2405.09605">
              Elements of World Knowledge (EWoK)
            </a>{" "}
            leaderboard in collaboration with the EWoK team. EWoK is a benchmark
            for evaluating world modeling in language models by testing their
            ability to use knowledge of a concept to match a target text with a
            plausible/implausible context. EWoK targets specific concepts from
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
