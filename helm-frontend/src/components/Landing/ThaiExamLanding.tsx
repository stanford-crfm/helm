import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function ThaiExamLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">ThaiExam</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p>
            In collaboration with{" "}
            <a
              href="https://www.scbx.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              SCBX
            </a>{" "}
            and{" "}
            <a
              href="https://www.scb10x.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              SCB 10X
            </a>
            , we introduce the ThaiExam HELM leaderboard. ThaiExam is a Thai
            language benchmark based on examinations for high school students
            and investment professionals in Thailand. The ThaiExam leaderboard
            is the first public leaderboard for large language models on Thai
            language scenarios, and features evaluations of leading language
            models. Like all other HELM leaderboards, the ThaiExam leaderboard
            provides full prompt-level transparency, and the results can be
            fully reproduced using the HELM framework. We hope that this
            leaderboard will encourage further work in multilingual language
            model evaluation.
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
