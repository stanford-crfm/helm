export default function MMLULanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        Massive Multitask Language Understanding (MMLU) on HELM
      </h1>
      <p>
        <strong>Massive Multitask Language Understanding (MMLU)</strong>{" "}
        <a href="https://arxiv.org/pdf/2009.03300.pdf" className="link">
          (Hendrycks et al, 2020)
        </a>{" "}
        is a multiple-choice question answering test that covers 57 tasks
        including elementary mathematics, US history, computer science, law, and
        more. We publish evaluation results from evaluating various models on
        MMLU using HELM. Our evaluation results include the following:
      </p>
      <ul className="my-2 list-disc list-inside">
        <li>Standarized prompting templates for comparability</li>
        <li>Break down of accuracy results for each of 57 subjects</li>
        <li>Full transparency of all raw prompts and predictions</li>
      </ul>
    </div>
  );
}
