import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function MMLUWinograndeAfrLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        MMLU-Winogrande-Afr: Clinical MMLU and Winogrande in 11 low-resource
        African languages
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="mb-4 italic">
            This leaderboard is a collaboration with{" "}
            <a
              href="https://ghamut.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              Ghamut Corporation
            </a>{" "}
            and the{" "}
            <a
              href="https://www.gatesfoundation.org/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              Gates Foundation
            </a>
            .
          </p>
          <p className="my-4">
            Large Language Models (LLMs) have shown remarkable performance
            across various tasks, yet significant disparities remain for
            non-English languages, and especially native African languages. This
            paper addresses these disparities by creating approximately 1
            million human-translated words of new benchmark data in 8
            low-resource African languages, covering a population of over 160
            million speakers of: Amharic, Bambara, Igbo, Sepedi (Northern
            Sotho), Shona, Sesotho (Southern Sotho), Setswana, and Tsonga. Our
            benchmarks are translations of Winogrande and three sections of
            MMLU: college medicine, clinical knowledge, and virology. Using the
            translated benchmarks, we report previously unknown performance gaps
            between state-of-the-art (SOTA) LLMs in English and African
            languages. The publicly available benchmarks, translations, and code
            from this study support further research and development aimed at
            creating more inclusive and effective language technologies.
          </p>

          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://arxiv.org/abs/2412.12417"
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
