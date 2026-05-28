import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function ArabicEnterpriseLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">
        HELM Arabic Enterprise
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-[1] text-l">
          <p className="my-4">
            We present HELM Arabic Enterprise, a leaderboard for transparent,
            reproducible evaluation of large language models on Arabic-language
            benchmarks designed around enterprise use cases. The leaderboard was
            developed in collaboration with{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://arabic.ai/"
            >
              Arabic.AI
            </a>
            .
          </p>

          <p className="my-4">
            Arabic enterprise applications often require more than general
            conversational ability. Models must generate grounded content,
            reason over financial concepts, answer domain-specific legal
            questions, and operate reliably in Arabic across formal,
            professional, and institutional registers. HELM Arabic Enterprise
            evaluates these capabilities through six tasks across content
            generation, financial reasoning, and legal question answering:
          </p>

          <ul className="list-disc pl-6">
            <li>Article Generation</li>
            <li>Financial Multiple Choice Question Anwering</li>
            <li>Financial Boolean Verification</li>
            <li>Financial Calculation</li>
            <li>Legal Open-book Question Answering</li>
            <li>Legal Closed-book Question Answering</li>
          </ul>

          <p className="my-4">
            As with all HELM leaderboards, HELM Arabic Enterprise emphasizes
            transparency and reproducibility. All model requests, responses,
            prompts, metrics, and scores are made available for inspection.
            Results can be reproduced using the{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://github.com/stanford-crfm/helm/"
            >
              open-source HELM framework
            </a>
            , allowing researchers and practitioners to audit model behavior
            rather than relying only on aggregate scores. We hope HELM Arabic
            Enterprise becomes a useful resource for the Arabic NLP community
            and for organizations evaluating LLMs for Arabic enterprise
            applications.
          </p>

          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://crfm.stanford.edu/2026/05/26/helm-arabic-enterprise.html"
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
