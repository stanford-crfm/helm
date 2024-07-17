import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import scb10x from "@/assets/logos/scb10x.png";

export default function MMLULanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">ThaiExam</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <div>
            <a href="https://scb10x.com/">
              <img
                src={scb10x}
                alt="Logo"
                className="mx-auto block my-4 w-48"
              />
            </a>
          </div>
          <p>
            In collaboration with{" "}
            <a
              href="https://www.scb10x.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              SCB 10X
            </a>
            , we introduce the HELM <span className="font-bold">ThaiExam</span>{" "}
            leaderboard. ThaiExam is a Thai language benchmark based on
            examinations for high-school students and investment professionals
            in Thailand. We present a leaderbaord of evaluations of leading
            language models on ThaiExam, which provide insights into the Thai
            language competency of these models. We hope that this leaderboard
            will encourage further work in multi-lingual language model
            developmpent and evaluation.
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
