import MiniLeaderboard from "@/components/MiniLeaderboard";

export default function MMLULanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        Massive Multitask Language Understanding (MMLU) on HELM
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <p>
            <strong>Massive Multitask Language Understanding (MMLU)</strong>{" "}
            <a href="https://arxiv.org/pdf/2009.03300.pdf" className="link">
              (Hendrycks et al, 2020)
            </a>{" "}
            is a multiple-choice question answering test that covers 57 tasks
            including elementary mathematics, US history, computer science, law,
            and more. We publish evaluation results from evaluating various
            models on MMLU using HELM. Our evaluation results include the
            following:
          </p>
          <ul className="my-2 list-disc list-inside">
            <li>Simple, standardized prompts</li>
            <li>Accuracy breakdown for each of the 57 subjects</li>
            <li>Full transparency of all raw prompts and predictions</li>
          </ul>
          <div className="flex flex-row justify-center mt-8">
            <a className="px-10 btn rounded-md" href="#/leaderboard">
              Full Leaderboard
            </a>
          </div>
        </div>
        <MiniLeaderboard numModelsToAutoFilter={10} />
      </div>
    </div>
  );
}
