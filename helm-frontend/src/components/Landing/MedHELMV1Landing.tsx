import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import overview from "@/assets/medhelm/medhelm-v1-overview.png";

export default function MedHELMLanding() {
  const benchmarkName = <strong className="font-bold">MedHELM</strong>;
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-4 font-bold text-center">MedHELM</h1>
      <p className="text-xl my-4 italic text-center">
        Medical and AI experts build a benchmark for evaluation of LLMs grounded
        in real-world healthcare needs.
      </p>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="my-4">
            We introduce {benchmarkName}, a comprehensive healthcare benchmark
            to evaluate language models on real-world clinical tasks using real
            electronic health records. Building on the HELM framework,{" "}
            {benchmarkName} comprises a structured taxonomy with 5 categories,
            22 subcategories, and 121 distinct clinical tasks as well as 31
            diverse datasets (12 private, 6 gated-access, and 13 public). The
            datasets represent a spectrum of healthcare scenarios, from
            diagnostic decision-making to patient communication, providing a
            more nuanced and clinically relevant assessment of AI capabilities
            in healthcare settings.
          </p>
          <img
            src={overview}
            alt="MedHELM Task Categories"
            className="mx-auto my-4 block w-7/8"
            sizes="100vw"
          />
          <p className="my-4">
            {benchmarkName} establishes a foundation for testing and evaluation
            of the real-world applicability of language models in healthcare. It
            is made possible by a unique collaboration between the Center for
            Research on Foundation Models, Technology and Digital Solutions at
            Stanford Healthcare, and Microsoft Healthcare and Life Sciences in
            partnership with faculty in the Departments of Medicine, Computer
            Science, Anesthesiology, Dermatology, Pediatrics and Biomedical Data
            Science as well as trainees from the MCiM program at the Clinical
            Excellence Research Center. The effort is coordinated by the Center
            for Biomedical Informatics Research.
          </p>
          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://arxiv.org/abs/2505.23802"
            >
              Paper
            </a>
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://crfm-helm.readthedocs.io/en/latest/medhelm/"
            >
              Documentation
            </a>
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://hai.stanford.edu/news/holistic-evaluation-of-large-language-models-for-medical-applications"
            >
              Blog Post
            </a>
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
              Leaderboard
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
