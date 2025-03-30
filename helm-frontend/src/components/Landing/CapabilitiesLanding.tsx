import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function CapabilitiesLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">HELM Capabilities</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p>
            Building on the HELM framework, we introduce{" "}
            <strong className="font-bold">HELM Capabilities</strong> to capture
            our latest thinking on the evaluation of general capabilities. HELM
            Capabilities is a new benchmark and leaderboard that consists of a
            curated set of scenarios for measuring various capabilities of
            language models. Like all other HELM leaderboards, the HELM
            Capabilities leaderboard provides full prompt-level transparency,
            and the results can be fully reproduced using the HELM framework.
          </p>
          <div className="flex flex-row justify-center my-4">
            <a
              href="https://crfm.stanford.edu/2025/03/20/helm-capabilities.html"
              className="px-10 btn rounded-md mx-4"
            >
              Blog Post
            </a>
            <Link to="leaderboard" className="px-10 btn rounded-md mx-4">
              Leaderboard
            </Link>
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
