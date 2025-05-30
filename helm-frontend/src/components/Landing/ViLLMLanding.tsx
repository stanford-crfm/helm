import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function ViLLMLanding() {
  const benchmarkName = <strong className="font-bold">ViLLM</strong>;
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        ViLLM: Crossing Linguistic Horizon
      </h1>
      <p className="text-xl my-4 italic text-center">
        {benchmarkName} is a comprehensive benchmark suite for evaluating the
        performance of language models in <strong>Vietnamese</strong>.
      </p>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            As multilingual large language models (LLMs) continue to advance
            natural language processing, bridging communication across diverse
            cultures and languages, their effectiveness in lower-resourced
            languages like Vietnamese remains limited. Despite being trained on
            large multilingual corpora, most open-source LLMs struggle with
            Vietnamese understanding and generation.
            <strong> ViLLM</strong> addresses this gap by providing a robust
            evaluation framework tailored specifically for Vietnamese. It
            includes <strong>11 essential scenarios</strong>, each targeting a
            core capability of Vietnamese LLMs:
          </p>

          <p className="my-4">
            <strong>ViLLM</strong> includes 11 carefully designed evaluation
            scenarios, each addressing a core language modeling capability:
            <ul className="list-disc list-inside mt-2">
              <li>
                <strong>Question Answering:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/juletxara/xquad_xtreme"
                >
                  XQuAD
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/facebook/mlqa"
                >
                  MLQA
                </a>
              </li>
              <li>
                <strong>Summarization:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/Yuhthe/vietnews"
                >
                  VietNews
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/GEM/wiki_lingua"
                >
                  WikiLingua
                </a>
              </li>
              <li>
                <strong>Sentiment Analysis:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/vlsp2016"
                >
                  VLSP2016
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/UIT-VSFC"
                >
                  UiT-VSFC
                </a>
              </li>
              <li>
                <strong>Text Classification:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/PhoATIS"
                >
                  PhoATIS
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/UIT-VSMEC"
                >
                  UiT-VSMEC
                </a>
              </li>
              <li>
                <strong>Knowledge:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/zalo_e2eqa"
                >
                  ZaloE2E
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/ViMMRC"
                >
                  UiT-ViMMRC
                </a>
              </li>
              <li>
                <strong>Toxic Detection:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD"
                >
                  UiT-VIHSD
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/tarudesu/ViCTSD"
                >
                  UiT-ViCTSD
                </a>
              </li>
              <li>
                <strong>Information Retrieval:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/unicamp-dl/mmarco"
                >
                  mMARCO
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/unicamp-dl/mrobust"
                >
                  mRobust04
                </a>
              </li>
              <li>
                <strong>Language Modeling:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/MLQA"
                >
                  MLQA
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/VSEC"
                >
                  VSEC
                </a>
              </li>
              <li>
                <strong>Reasoning:</strong>{" "}
                <a className="link-primary" href="">
                  Synthetic reasoning
                </a>
                ,{" "}
                <a className="link-primary" href="">
                  Natural synthetic reasoning
                </a>
              </li>
              <li>
                <strong>Mathematic:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/Vietnamese-MATH"
                >
                  MATH
                </a>
              </li>
              <li>
                <strong>Translation:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/vietgpt/opus100_envi"
                >
                  OPUS100
                </a>
                ,{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/ura-hcmut/PhoMT"
                >
                  PhoMT
                </a>
              </li>
            </ul>
          </p>

          <p className="my-4">
            <strong>ViLLM</strong> also includes tools to promote the ethical
            and responsible use of language models:
            <ul className="list-disc list-inside mt-2">
              <li>
                <strong>Bias Assessment:</strong> Detects and mitigates biased
                patterns in model outputs.
              </li>
              <li>
                <strong>Toxicity Assessment:</strong> Monitors and controls the
                generation of harmful or offensive content.
              </li>
              <li>
                <strong>Fairness Evaluation:</strong> Ensures equitable
                performance across demographic groups and languages.
              </li>
              <li>
                <strong>Robustness Analysis:</strong> Evaluates model stability
                against noisy or adversarial inputs in real-world scenarios.
              </li>
            </ul>
          </p>
          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://aclanthology.org/2024.findings-naacl.182"
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
