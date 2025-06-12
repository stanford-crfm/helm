import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function LongContextLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">HELM Long Context</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-[1] text-l">
          <p className="my-4">
            Recent Large Language Models (LLMs) claim to have powerful
            long-context capabilities (i.e. the ability to process and reason
            over long inputs), but these claims have not been rigorously
            evaluated. To address this, we introduce the{" "}
            <strong className="font-bold">HELM Long Context</strong>{" "}
            leaderboard, which evaluates leading long context LLMs on a broad
            set of long context tasks.
          </p>
          <p className="my-4">
            The benchmark consists of the following tasks: question answering
            over a set of retrieved passages (
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="#/leaderboard/ruler_hotpotqa"
            >
              RULER HotPotQA
            </a>
            ,{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="#/leaderboard/ruler_squad"
            >
              RULER SQuAD
            </a>
            ), question answering over a long passage (
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="#/leaderboard/infinite_bench_en_qa"
            >
              ∞Bench En.QA
            </a>
            ), summarization of a long passage (
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="#/leaderboard/infinite_bench_en_sum"
            >
              ∞Bench En.Sum
            </a>
            ), and multi-round co-reference resolution over a long multi-turn
            conversation (
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="#/leaderboard/openai_mrcr"
            >
              OpenAI MRCR
            </a>
            ). The leaderboard demonstrates that even though significant
            progress has been made on long context capabilities, there is still
            considerable room for improvement.
          </p>
          <p className="my-4">
            As with all HELM leaderboards, this leaderboard provides full
            transparency into all LLM requests and responses, and the results
            are reproducible using the HELM open source framework.
          </p>
          <p className="my-4">
            This leaderboard was produced through research collaboration with{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://www.lvmh.com/"
            >
              LVMH
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
          <div className="flex flex-row justify-center mt-4">
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
              Full Leaderboard
            </a>
          </div>
        </div>
        <div
          className="flex-[1] py-2 rounded-3xl bg-gray-100 h-full" // Stretched to full height
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
