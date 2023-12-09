import helmHero from "@/assets/helmhero.png";
import { Link } from "react-router-dom";
import MiniLeaderboard from "./MiniLeaderboard";

export default function Hero() {
  return (
    <div className="flex flex-col px-6 py-14">
      {/* Text section */}
      <div className="text-center mb-6">
        <h1 className="text-4xl mb-4 mx-2 mt-2">
          <strong>
            A holistic framework for evaluating foundation models.
          </strong>
        </h1>
      </div>

      {/* Image section */}
      <div className="flex flex-row justify-center">
        {/* Left side content */}
        <div className="w-full md:w-1/2">
          <div className="flex justify-center">
            <div className="flex justify-center">
              <img
                src={helmHero}
                alt="HELM Hero"
                className="object-cover w-full h-full"
                style={{ width: "480px", height: "456px" }}
              />
            </div>
          </div>
        </div>

        {/* Right side content */}
        <div className="w-full md:w-1/2">
          <div className="flex justify-center">
            <div className="py-2 rounded-3xl bg-gray-100">
              <MiniLeaderboard></MiniLeaderboard>
              <div className="flex justify-end mt-2 ">
                <Link to="leaderboard">
                  <button className="px-4 mx-3 my-1 btn bg-white rounded-md">
                    <body>See More</body>
                  </button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
