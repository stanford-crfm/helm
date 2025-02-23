import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import overview from "@/assets/medhelm/medhelm-overview.png";

export default function MedHELMLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">MedHELM</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <img
            src={overview}
            alt="MedHELM Task Categories"
            className="mx-auto my-4 block w-3/4"
            sizes="100vw"
          />
          We introduce <strong className="font-bold">MedHELM</strong>, a
          comprehensive healthcare benchmark to evaluate language models on
          real-world clinical tasks using real electronic health records.
          Building on the HELM framework, MedHELM comprises a structured
          taxonomy with 5 categories, 22 subcategories, and 121 distinct
          clinical tasks as well as 31 diverse datasets (12 private, 6
          gated-access, and 13 public), representing a spectrum of healthcare
          scenarios, from diagnostic decision-making to patient communication,
          providing a more nuanced and clinically relevant assessment of AI
          capabilities in healthcare settings. MedHELM establishes a foundation
          for testing and evaluation of the real-world applicability of language
          models in healthcare.
          <div className="flex flex-row justify-center mt-4">
            <a className="px-10 btn rounded-md mx-4" href="#">
              Blog Post
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
