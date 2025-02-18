import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import aisingapore from "@/assets/logos/aisingapore.png";

export default function BhasaLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">Bhasa</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <div className="text-center">
            <a href="https://aisingapore.org/">
              <img src={aisingapore} alt="Logo" className="inline h-32 mx-4 my-4" />
            </a>
          </div>
          <p>
            <a
              href="https://aisingapore.org/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              AI Singapore
            </a>
            introduces the Bhasa leaderboard. Bhasa is an assessment of
            large language models across various tasks for Southeast Asian
            languages. The leaderboard evaluates four key multilingual
            capabilities for language models: performance on natural language
            understanding (NLU), natural language generation (NLG), natural
            language reasoning (NLR), and linguistic diagnostics.
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
