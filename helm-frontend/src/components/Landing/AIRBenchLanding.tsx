import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function MMLULanding() {
  const benchNameStyle = {
    fontVariant: "small-caps",
    fontWeight: "bold",
  };
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">
        <span style={benchNameStyle}>AIRbench</span> 2024
      </h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p>
            We introduce <span style={benchNameStyle}>AIRbench</span>, the first
            AI safety benchmark that aligns with a vast of emerging government
            regulations and company policies, following the regulation-based
            safety categories grounded in our AI Risks study, AIR 2024. AIR 2024
            decomposes government regulations and company policies into a
            four-tiered safety taxonomy with granular risk categories in the
            lowest tier.
            <span style={benchNameStyle}>AIRbench</span> contains diverse
            prompts spanning these categories, with manual curation and human
            auditing to ensure quality. We evaluate leading models on{" "}
            <span style={benchNameStyle}>AIRbench</span>, uncovering insights
            into their alignment with regulation or policy-specified safety
            concerns. By bridging the gap between public benchmarks and
            practical AI risks using the unified language of risk
            categorizations, <span style={benchNameStyle}>AIRbench</span>{" "}
            provides a foundation for assessing model safety across
            jurisdictions, fostering the development of safer and more
            responsible AI systems.
          </p>
          <div className="flex flex-row justify-center mt-8">
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
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
