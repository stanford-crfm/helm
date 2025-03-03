import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function ToRRLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        The Mighty ToRR: A Benchmark for Table Reasoning and Robustness
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="mb-4 italic">
            This leaderboard is a collaboration with{" "}
            <a
              href="https://research.ibm.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              IBM Research
            </a>
            .
          </p>
          <p className="my-4">
            Despite its real-world significance, model performance on tabular
            data remains underexplored, leaving uncertainty about which model to
            rely on and which prompt configuration to adopt. To address this
            gap, we create <strong className="font-bold">ToRR</strong>, a
            benchmark for Table Reasoning and Robustness, a varied benchmark
            that measures model performance and robustness on table-related
            tasks. The benchmark includes 10 datasets that cover different types
            of table reasoning capabilities across varied domains. ToRR goes
            beyond model performance rankings, and is designed to reflect
            whether models can handle tabular data consistently and robustly,
            across a variety of common table representation formats. We present
            a leaderboard as well as comprehensive analyses of the results of
            leading models over ToRR. Our results reveal a striking pattern of
            brittle model behavior, where even strong models are unable to
            perform robustly on tabular data tasks. Although no specific table
            format leads to consistently better performance, we show that
            testing over multiple formats is crucial for reliably estimating
            model capabilities. Moreover, we show that the reliability boost
            from testing multiple prompts can be equivalent to adding more test
            examples. Overall, our findings show that reasoning over table tasks
            remains a significant challenge.
          </p>

          <div className="flex flex-row justify-center mt-4">
            <a className="px-10 btn rounded-md mx-4 hidden" href="#">
              Paper
            </a>
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
              Full Leaderboard
            </a>
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
