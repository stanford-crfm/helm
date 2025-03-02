import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function SEAHELMLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        SEA-HELM: Southeast Asian Holistic Evaluation of Language Models
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          With the rapid emergence of novel capabilities in Large Language
          Models (LLMs), the need for rigorous multilingual and multicultural
          benchmarks that are integrated has become more pronounced. Though
          existing LLM benchmarks are capable of evaluating specific
          capabilities of LLMs in English as well as in various mid- to
          low-resource languages, including those in the Southeast Asian (SEA)
          region, a comprehensive and authentic evaluation suite for the SEA
          languages has not been developed thus far. Here, we present{" "}
          <strong className="font-bold">SEA-HELM</strong>, a holistic linguistic
          and cultural LLM evaluation suite that emphasizes SEA languages,
          comprising five core pillars: (1) NLP Classics, (2) LLM-specifics, (3)
          SEA Linguistics, (4) SEA Culture, (5) Safety. SEA-HELM currently
          supports Filipino, Indonesian, Tamil, Thai, and Vietnamese. We also
          introduce the SEA-HELM leaderboard, which allows users to understand
          models' multilingual and multicultural performance in a systematic and
          user-friendly manner.
          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://arxiv.org/abs/2502.14301"
            >
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
