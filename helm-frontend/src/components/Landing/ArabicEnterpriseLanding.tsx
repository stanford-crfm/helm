import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function ArabicEnterpriseLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">HELM Arabic</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-[1] text-l">
          <p className="my-4">
            We present{" "}
            <strong className="font-bold">HELM Arabic Enterpise</strong>, a
            leaderboard for transparent and reproducible evaluation of LLMs on
            Arabic language benchmarks for enterprise use cases. This
            leaderboard was created in collaboration with{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://arabic.ai/"
            >
              Arabic.AI
            </a>
            .
          </p>

          <p className="my-4">
            HELM Arabic Enterprise introduces new datasets for the following
            tasks:
          </p>

          <ul className="list-disc pl-6">
            <li>
              <strong className="font-bold">Content Generation</strong> &mdash;
              Given summaries of from real news articles, the LLM is prompted to
              generate new articles in a corporate style. The generated articles
              are then graded by an LLM-as-judge for faithfulness, completeness,
              and style adherence.
            </li>
            <li>
              <strong className="font-bold">Financial</strong> &mdash; Given
              questions from English language finance textbooks that have been
              translated to Arabic using machine translation, the LLM is
              prompted to answer the questions in one of the following three
              settings. In the MCQ setting, the LLM must choose between three
              possible choices. In the boolean setting, the LLM must decide if a
              given statement is true or false. In the calculation setting, the
              LLM must perform mathematical calculations to produce a numeric
              answer.
            </li>
            <li>
              <strong className="font-bold">Legal</strong> &mdash; Given
              open-ended questions written by Arabic legal experts, the LLM is
              prompted to respond with short answers to the questions in one of
              the following two settings. In the RAG setting, the LLM is also
              provided with the relevant legal statute in the prompt. In the QA
              setting, the LLM is not provided with the relevant legal statute,
              and must answer relying only on its domain knowledge.
            </li>
          </ul>

          <p className="my-4">
            The{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://crfm.stanford.edu/helm/arabic-enterprise/latest/"
            >
              leaderboard results
            </a>{" "}
            show a wide range of scores on these tasks. As with all HELM
            leaderboards, this leaderboard provides full transparency into all
            LLM requests and responses, and the results are{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://crfm-helm.readthedocs.io/en/latest/reproducing_leaderboards/"
            >
              reproducible
            </a>{" "}
            using the{" "}
            <a
              className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://github.com/stanford-crfm/helm/"
            >
              HELM open source framework
            </a>
            . We hope that this leaderboard will be a valuable resource for the
            Arabic NLP community.
          </p>

          <div className="flex flex-row justify-center mt-4">
            <a className="px-10 btn rounded-md mx-4" href="#">
              Blog Post (Upcoming)
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
