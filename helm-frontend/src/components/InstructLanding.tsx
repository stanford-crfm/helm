export default function InstructLanding() {
  // TODO: Link to blog post.
  return (
    <div className="mx-auto text-lg px-16">
      <div className="container mb-12 mx-auto text-lg px-16">
        <h1 className="text-4xl mt-40">HELM-Instruct</h1>
        <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-8 md:gap-32 my-4">
          <a
            className="px-10 btn rounded-md"
            href="https://arxiv.org/pdf/2211.09110.pdf"
          >
            Blog Post
          </a>
          <a
            className="px-10 btn rounded-md"
            href="https://github.com/stanford-crfm/helm"
          >
            Github
          </a>
        </div>
        <p>
          We introduce HELM-Instruct, a multidimensional and absolute-score
          evaluation framework for instruction-following LLMs. We use
          HELM-Instruct to compare several instruction-following models as well
          as some Human- and LLM- evaluators.
        </p>
      </div>
    </div>
  );
}
