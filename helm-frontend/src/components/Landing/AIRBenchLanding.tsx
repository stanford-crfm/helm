import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";
import overview from "@/assets/air-bench/air-overview.png";

export default function MMLULanding() {
  const benchNameStyle = {
    fontVariant: "small-caps",
    fontWeight: "bold",
  };
  const benchName = <span style={benchNameStyle}>AIR-Bench 2024</span>;
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">{benchName}</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <img
            src={overview}
            alt="AIR 2024 Categories"
            className="mx-auto my-4 block w-3/4"
            sizes="100vw"
          />
          <p>
            We introduce {benchName}, the first AI safety benchmark aligned with
            emerging government regulations and company policies, following the
            regulation-based safety categories grounded in our AI Risks study,
            AIR 2024. AIR 2024 decomposes 8 government regulations and 16
            company policies into a four-tiered safety taxonomy with 314
            granular risk categories in the lowest tier. {benchName} contains
            5,694 diverse prompts spanning these categories, with manual
            curation and human auditing to ensure quality. We evaluate leading
            language models on {benchName}, uncovering insights into their
            alignment with specified safety concerns. By bridging the gap
            between public benchmarks and practical AI risks, {benchName}{" "}
            provides a foundation for assessing model safety across
            jurisdictions, fostering the development of safer and more
            responsible AI systems.
          </p>
          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="#/leaderboard"
              style={{ display: "none" }}
            >
              Paper (TBD)
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
          <MiniLeaderboard numModelsToAutoFilter={10} />
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
