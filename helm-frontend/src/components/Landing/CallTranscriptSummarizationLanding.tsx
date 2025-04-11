import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function CallTranscriptSummarizationLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">
        HELM Call Transcript Summarization Leaderboard
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            Large language models (LLMs) show great potential for call center
            applications, yet there is a lack of domain-specific and
            ecologically valid evaluations in this domain. To address this, we
            introduce the{" "}
            <strong className="font-bold">
              HELM Call Transcript Summarization
            </strong>{" "}
            leaderboard, which evaluates leading LLMs on a summarization task
            over a dataset of real call transcripts provided by Accenture.
          </p>
          <p className="my-4">
            This dataset consists of 162 transcribed calls to an internal
            corporate IT helpdesk. The calls were transcribed using an automatic
            speech recognition (ASR) model. Transcription errors were
            deliberately left uncorrected to reflect the nature of real-life
            transcripts. The transcripts were anonymized using a semi-automated
            process with human verification.
          </p>
          <p className="my-4">
            To evaluate the LLMs, summaries of the transcripts were generated
            using 17 LLMs. The quality of the generated summaries were then
            evaluated using LLM-as-judge with an ensemble of 3 models.
          </p>
          <p className="my-4">
            As with all HELM leaderboards, this leaderboard provides full
            transparency into all LLM requests and responses, and the results
            are reproducible using the HELM open source framework. We hope that
            this leaderboard offers initial insights into the potential of LLMs
            for this task.
          </p>
          <p className="my-4">
            This leaderboard was produced through research collaboration with{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://www.accenture.com/"
            >
              Accenture
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
