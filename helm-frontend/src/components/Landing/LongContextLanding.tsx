import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function LongContextLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">HELM Long Context</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-[1] text-l">
          <p className="my-4">
            Recent Large Language Models (LLMs) support processing long inputs
            with hundreds of thousands or millions of tokens. Long context
            capabilities are important for many real-world applications, such as
            processing long text documents, conducting long conversations or
            following complex instructions. However, support for long inputs
            does not equate to strong long context capabilities. As such, there
            is a need for rigorous and comprehensive evaluations of long context
            capabilities.
          </p>
          <p className="my-4">
            To address this, we introduce the{" "}
            <strong className="font-bold">HELM Long Context</strong>{" "}
            leaderboard, which provides transparent, comparable and reproducible
            evaluations of long context capabilities of recent models. The
            benchmark consists of 5 tasks:
          </p>
          <ul className="list-disc pl-6">
            <li>
              <a
                className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
                href="#/leaderboard/ruler_squad"
              >
                <strong className="font-bold">RULER SQuAD</strong>
              </a>{" "}
              &mdash; open ended single-hop question answering on passages
            </li>
            <li>
              <a
                className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
                href="#/leaderboard/ruler_hotpotqa"
              >
                <strong className="font-bold">RULER HotPotQA</strong>
              </a>{" "}
              &mdash; open ended multi-hop question answering on passages
            </li>
            <li>
              <a
                className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
                href="#/leaderboard/infinite_bench_en_qa"
              >
                <strong className="font-bold">∞Bench En.MC</strong>
              </a>{" "}
              &mdash; multiple choice question answering based on the plot of a
              novel
            </li>
            <li>
              <a
                className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
                href="#/leaderboard/infinite_bench_en_sum"
              >
                <strong className="font-bold">∞Bench En.Sum</strong>
              </a>{" "}
              &mdash; summarization of the plot of a novel
            </li>
            <li>
              <a
                className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
                href="#/leaderboard/openai_mrcr"
              >
                <strong className="font-bold">OpenAI MRCR</strong>
              </a>{" "}
              &mdash; multi-round co-reference resolution on a long, multi-turn,
              synthetic conversation
            </li>
          </ul>

          <p className="my-4">
            The results demonstrate that even though significant progress has
            been made on long context capabilities, there is still considerable
            room for improvement.
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
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://crfm.stanford.edu/2025/09/29/helm-long-context.html"
            >
              Blog Post
            </a>
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
