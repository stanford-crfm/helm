import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

import accenture from "@/assets/logos/accenture.png";
import cresta from "@/assets/logos/cresta.png";

export default function CallCenterLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">HELM Call Center</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <div className="text-center">
            <a href="https://www.accenture.com/">
              <img
                src={accenture}
                alt="Logo"
                className="inline h-12 mx-4 my-4"
              />
            </a>
            <a href="https://www.cresta.com/">
              <img src={cresta} alt="Logo" className="inline h-8 mx-4 my-4" />
            </a>
          </div>
          <p>
            In collaboration with{" "}
            <a
              href="https://www.accenture.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              Accenture
            </a>{" "}
            and{" "}
            <a
              href="https://www.cresta.com/"
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
            >
              Cresta
            </a>
            , we introduce the HELM{" "}
            <span className="font-bold">Call Center</span> leaderboard. HELM
            Call Center is a leaderboard consisting of evaluations of leading
            language models on scenarios with realistic tasks from the call
            center context.
          </p>
          <div className="flex flex-row justify-center my-4">
            <Link to="leaderboard" className="px-10 btn rounded-md mx-4">
              Full Leaderboard
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
