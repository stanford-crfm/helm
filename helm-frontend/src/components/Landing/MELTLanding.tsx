import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function MELTLanding() {
  const benchmarkName = <strong className="font-bold">MELT</strong>;
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">MELT</h1>
      <p className="text-xl my-4 italic text-center">
        {benchmarkName} is collection of benchmarks for evaluating language
        models in Vietnamese.
      </p>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            The recent emergence of multilingual large language models (LLMs) is
            revolutionizing natural language processing, bridging communication
            gaps across diverse cultures and languages. However, to truly
            harness the potential of these models, it's crucial to understand
            their strengths and limitations across a wide range of languages and
            tasks.
            {benchmarkName} is designed with this in mind, offering a
            comprehensive approach to evaluate LLMs in various linguistic
            contexts. Recognizing that proficiency in one language or task does
            not guarantee similar performance elsewhere, {benchmarkName} enables
            users to pinpoint specific areas for improvement, fostering the
            development of robust and reliable multilingual language
            technologies.
          </p>
          <p className="my-4">
            {benchmarkName} includes ten carefully selected evaluation
            scenarios, each targeting a key aspect of LLM capability:
            <ul className="list-disc list-inside">
              <li>
                Summarization: Evaluates the model's ability to condense large
                texts while retaining essential information.
              </li>
              <li>
                Question-Answering: Assesses comprehension and accurate
                extraction of answers from provided contexts.
              </li>
              <li>
                Knowledge: Tests the model's ability to recall and apply
                information across different domains.
              </li>
              <li>
                Sentiment Analysis: Measures the ability to detect and classify
                emotional tones in text.
              </li>
              <li>
                Text Classification: Evaluates accuracy in categorizing text
                into predefined labels.
              </li>
              <li>
                Toxic Detection: Identifies the model's capacity to flag harmful
                or biased language.
              </li>
              <li>
                Language Modeling: Tests fluency and coherence in generating
                contextually appropriate text.
              </li>
              <li>
                Reasoning: Measures logical deduction and understanding of
                complex relationships.
              </li>
              <li>
                Math: Assesses competency in solving mathematical problems in
                text form.
              </li>
              <li>
                Information Retrieval: Tests effectiveness in searching,
                retrieving, and synthesizing relevant information.
              </li>
            </ul>
          </p>
          <p className="my-4">
            {benchmarkName} also includes tools to ensure the ethical deployment
            of LLMs:
            <ul className="list-disc list-inside">
              <li>
                Bias Assessment: Identifies and mitigates potential biases in
                model outputs.
              </li>
              <li>
                Toxicity Assessment: Measures and controls the generation of
                harmful or offensive language.
              </li>
              <li>
                Fairness Evaluation: Ensures equitable performance across
                demographic groups and languages.
              </li>
              <li>
                Robustness Analysis: Examines resilience to noisy inputs and
                adversarial attacks, ensuring reliable performance in real-world
                scenarios.
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
