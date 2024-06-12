import MiniLeaderboard from "@/components/MiniLeaderboard";
import overview from "@/assets/image2struct/overview.png";
import processFlow from "@/assets/image2struct/process-flow.png";

export default function Image2StructLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        Image2Struct: A Benchmark for Evaluating Vision-Language Models in
        Extracting Structured Information from Images
      </h1>
      <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-2 md:gap-8 my-8">
        <a className="px-10 btn rounded-md" href="TODO">
          Paper
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://github.com/stanford-crfm/helm"
        >
          Github
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://huggingface.co/datasets/stanford-crfm/i2s-latex"
        >
          Latex dataset
        </a>
        <a
          className="px-5 btn rounded-md"
          href="https://huggingface.co/datasets/stanford-crfm/i2s-webpage"
        >
          Webpage dataset
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://huggingface.co/datasets/stanford-crfm/i2s-musicsheet"
        >
          Music sheet dataset
        </a>
      </div>

      <div className="flex flex-col lg:flex-row items-center gap-8">
        <div className="flex-1 text-xl">
          <p>
            <strong>Image2struct</strong> is a benchmark for evaluating
            vision-language models in practical tasks of extracting structured
            information from images.
          </p>
          <br />
          <p>
            In our tasks, VLMs are prompted to generate the underlying
            structured information (i.e., code) from an input image. The code
            can be compiled, and the output image is evaluated against the input
            image to produce a score. This round-trip evaluation allows us to
            quantitatively evaluate VLMs on complex tasks with multiple correct
            answers. We create a pipeline that downloads fresh, user-submitted
            data from active online communities upon execution, evaluates the
            VLMs shortly, and produces a leaderboard.
          </p>
          <br />
          <img
            src={overview}
            alt="Evaluation flowchart"
            className="mx-auto block w-full"
            sizes="100vw"
          />
          <br />
          <p>We introduce 3 tasks:</p>
          <ul className="my-2 list-disc list-inside">
            <li>
              LaTex: equations, tables, plots and algorithms form ArXiV papers
            </li>
            <li>
              Webpages: pages from GitHub written in HTML, CSS and Javascript,
              ...
            </li>
            <li>
              Music sheets: crops of measures from music sheets from IMSLP
            </li>
          </ul>
          <div className="flex flex-row justify-center mt-8">
            <a className="px-10 btn rounded-md" href="#/leaderboard">
              Full Leaderboard
            </a>
          </div>
        </div>
        <div className="flex-1">
          <MiniLeaderboard numModelsToAutoFilter={12} />
        </div>
      </div>
      <br />

      <div className="flex flex-col lg:flex-row items-center gap-8">
        <div className="flex-1 text-xl">
          <p>
            We provide an automated process for collecting new fresh data from
            online communities, evaluating the models and producing a
            leaderboard. The pipeline is designed to be executed on a regular
            basis to keep the leaderboard up-to-date.
          </p>
          <br />
          <p>
            In addition to the automated data collection, we also provide a{" "}
            <i>wild</i> subset for the LaTeX and webpage tasks that are
            collected from Wikipedia and various popular websites. These
            instances do not have a corresponding code, and the evaluation is
            done by our proposed metric: block EMD (Earth Mover Distance).
          </p>
        </div>
        <div className="flex-1">
          <img
            src={processFlow}
            alt="7 scenarios, 4 models, 4 evaluators and 5 criteria"
            className="mx-auto block w-full"
            sizes="200vw"
          />
        </div>
      </div>
    </div>
  );
}
