import helmHero from "@/assets/helmhero.png";
import { Link } from "react-router-dom";
import MiniLeaderboard from "./MiniLeaderboard";

export default function Hero() {
  return (
    <div className="flex flex-col px-4 sm:px-6 py-100 sm:py-10 sm:mb-96 md:mb-96 lg:mb-0 xl:mb-0 2xl:mb-0">
      {/* Text section */}
      <div className="flex flex-col text-center mb-10 justify-start">
        <h1 className="text-3xl sm:text-4xl mb-3 sm:mb-4 mx-2 mt-2">
          <strong>
            A holistic framework for evaluating foundation models.
          </strong>
        </h1>
      </div>

      {/* Container for Image and Leaderboard */}
      <div
        className="flex flex-col md:flex-col lg:flex-row lg:justify-center"
        style={{ height: "525px", transform: "scale(0.9)" }} // Reduced height by 10%
      >
        {/* Image section */}
        <div className="w-full lg:w-1/2 flex justify-center mb-4 lg:mb-0 h-full py-10">
          <img
            src={helmHero}
            alt="HELM Hero"
            className="object-cover h-full" // Stretched to full height
            style={{ maxWidth: "100%" }}
          />
        </div>

        {/* Leaderboard section */}
        <div className="w-full lg:w-1/2 flex justify-center h-full py-10">
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
    </div>
  );
}
