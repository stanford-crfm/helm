import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function ArabicLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">HELM Arabic</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-[1] text-l">



<p className="my-4">As part of our efforts to better understand the multilingual capabilities of large language models (LLMs), we present <strong className="font-bold">HELM Arabic</strong>, a leaderboard for transparent and reproducible evaluation of LLMs on Arabic language benchmarks. This leaderboard was produced in collaboration with and financially supported by <a className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://tarjama.com/">Tarjama</a>.</p>

<p className="my-4">HELM Arabic is based on <a className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://huggingface.co/blog/leaderboard-arabic-v2">version 2 of the Open Arabic LLM Leaderboard (OALL)</a>, a widely used leaderboard for Arabic LLMs introduced by Technology Innovation Institute (TII) in 2025. HELM Arabic uses the same 7 tasks from OALL:</p>

<ul className="list-disc pl-6">
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://aclanthology.org/2023.arabicnlp-1.21/">AlGhafa</a> &mdash; an Arabic language multiple choice evaluation benchmark derived from publicly available NLP datasets</li>
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://huggingface.co/datasets/MBZUAI/ArabicMMLU">ArabicMMLU</a> &mdash; a native Arabic language question answering benchmark using questions sourced from school exams across diverse educational levels in different countries spanning North Africa, the Levant, and the Gulf regions</li>
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://aclanthology.org/2020.emnlp-main.438/">Arabic EXAMS</a> &mdash; the Arabic language subset of the EXAMS multilingual question answering benchmark, which consists of high school exam questions across various school subjects</li>
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://huggingface.co/datasets/MBZUAI/MadinahQA">MadinahQA</a> &mdash; a question answering benchmark published by MBZUAI that tests knowledge of Arabic language and grammar</li>
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://arxiv.org/abs/2410.17040">AraTrust</a> &mdash; an Arab-region-specific safety evaluation dataset consisting of human-written questions including direct attacks, indirect attacks, and harmless requests with sensitive words</li>
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://huggingface.co/datasets/OALL/ALRAGE">ALRAGE</a> &mdash; an Arabic language passage-based open-ended model-graded question answering benchmark that reflects retrieval-augmented generation use cases</li>
<li><a className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://huggingface.co/datasets/MBZUAI/human_translated_arabic_mmlu">ArbMMLU-HT</a> &mdash; a translation of MMLU to Arabic by human translators published by MBZUAI</li>
</ul>

<p className="my-4">The <a className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://crfm.stanford.edu/helm/arabic/latest/">leaderboard results</a> show that LLMs have made significant progress in Arabic language understanding over the last few years. As with all HELM leaderboards, this leaderboard provides full transparency into all LLM requests and responses, and the results are <a className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://crfm-helm.readthedocs.io/en/latest/reproducing_leaderboards/">reproducible</a> using the <a className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600" href="https://github.com/stanford-crfm/helm/">HELM open source framework</a>. We hope that this leaderboard will be a valuable resource for the Arabic NLP community.</p>

      <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://crfm.stanford.edu/2025/12/16/helm-arabic.html"
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
