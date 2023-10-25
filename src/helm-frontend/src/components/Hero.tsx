import helmHero from "@/assets/helmhero.png";
import { Link } from "react-router-dom";

export default function Hero() {
  return (
    <div className="flex px-4 py-36">
      {/* Left side content */}
      <div className="flex-1 p-4 flex flex-col justify-center">
        {" "}
        {/* Added flex and justify-center */}
        <h1 className="text-4xl mb-4 mx-4">
          {" "}
          {/* Added mx-4 for horizontal margin */}
          HELM is{" "}
          <strong>
            {" "}
            a transparent benchmarking system for language models{" "}
          </strong>
          , providing standardized evaluations with multiple metrics and open
          access.
        </h1>
        <div className="flex justify-end w-1/4 ">
          <Link to="/leaderboard">
            <button className="px-10 btn btn-grey rounded-md">
              <body>Visit Leaderboard</body>
            </button>
          </Link>
        </div>
      </div>

      {/* Right side image */}
      <div className="w-1/3 mx-4">
        {" "}
        {/* Added mx-4 for horizontal margin */}
        <img
          src={helmHero}
          alt="HELM Hero"
          className="object-cover w-full h-full"
        />
      </div>
    </div>
  );
}
