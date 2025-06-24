import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function SLPHelmLanding() {
  const benchmarkName = <strong className="font-bold">SLPHelm</strong>;
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        SLPHelm: Advancing Speech Language Processing
      </h1>
      <p className="text-xl my-4 italic text-center">
        {benchmarkName} is a comprehensive benchmark suite for evaluating the
        performance of speech language models across multiple languages and
        tasks.
      </p>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            As speech language models continue to evolve and transform how we
            interact with technology, there is a growing need for standardized
            evaluation frameworks that can assess their capabilities across
            diverse languages and tasks. Despite significant advances in speech
            processing, many models struggle with multilingual understanding,
            accent recognition, and domain-specific speech tasks.{" "}
            <strong>SLPHelm</strong> addresses these challenges by providing a
            robust evaluation framework that includes
            <strong> 5 key scenarios</strong> across <strong>15 models</strong>,
            each targeting essential speech processing capabilities:
          </p>

          <p className="my-4">
            <strong>SLPHelm</strong> includes 5 carefully designed evaluation
            scenarios, each addressing a core speech processing capability:
            <ul className="list-disc list-inside mt-2">
              <li>
                <strong>Disorder Diagnosis</strong>
              </li>
              <li>
                <strong>Transcription Accuracy</strong>
              </li>
              <li>
                <strong>Disorder Type Diagnosis</strong>
              </li>
              <li>
                <strong>Disorder Symptom Diagnosis</strong>
              </li>
              <li>
                <strong>Disorder Diagnosis via Transcription</strong>
              </li>
            </ul>
          </p>

          <p className="my-4">
            <strong>SLPHelm</strong> evaluates models using comprehensive
            datasets:
            <ul className="list-disc list-inside mt-2">
              <li>
                <strong>UltraSuite:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/SAA-Lab/SLPHelmUltraSuite"
                >
                  UltraSuite
                </a>
              </li>
              <li>
                <strong>ENNI:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/SAA-Lab/SLPHelmDataset/tree/main/ENNI"
                >
                  ENNI
                </a>
              </li>
              <li>
                <strong>LeNormand:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/SAA-Lab/SLPHelmDataset/tree/main/LeNormand"
                >
                  LeNormand
                </a>
              </li>
              <li>
                <strong>PERCEPT-GFTA:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/SAA-Lab/SLPHelmDataset/tree/main/PERCEPT-GFTA"
                >
                  PERCEPT-GFTA
                </a>
              </li>
              <li>
                <strong>UltraSuite w/ Manual Labels:</strong>{" "}
                <a
                  className="link-primary"
                  href="https://huggingface.co/datasets/SAA-Lab/SLPHelmManualLabels"
                >
                  SLPHelmManualLabels
                </a>
              </li>
            </ul>
          </p>

          <div className="flex flex-row justify-center mt-4">
            <a className="px-10 btn rounded-md mx-4" href="#">
              Paper
            </a>
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
              Full Leaderboard
            </a>
          </div>
        </div>
        <div
          className="py-2 pb-6 rounded-3xl bg-gray-100 h-full"
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
