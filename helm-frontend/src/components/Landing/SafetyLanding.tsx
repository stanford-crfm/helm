import MiniLeaderboard from "@/components/MiniLeaderboard";
import helmSafety from "@/assets/helm-safety.png";

export default function SafetyLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 my-8 font-bold text-center">HELM Safety</h1>

      <div className="flex flex-col lg:flex-row items-center gap-8">
        <div className="flex-1 text-xl">
          <img
            src={helmSafety}
            alt="Logo"
            className="mx-auto p-0 block"
            style={{ width: "300px" }}
          />
          <p>
            Language models demonstrate powerful capabilities and pose
            significant risks. Given their widespread deployment, standardized
            public benchmarking of such models is vital. While language models
            are routinely evaluated on standard capability benchmarks,
            comparable standardization for benchmarking safety risks lags
            behind. To address this gap, we introduce HELM-Safety as a
            collection of 5 safety benchmarks that span 6 risk categories (e.g.
            violence, fraud, discrimination, sexual, harassment, deception). We
            present evaluation results for recent leading open weights and
            closed models.
          </p>
          <div className="flex flex-row justify-center mt-4">
            <a
              className="px-10 btn rounded-md mx-4"
              href="https://crfm.stanford.edu/2024/11/08/helm-safety.html"
            >
              Blog Post
            </a>
            <a className="px-10 btn rounded-md mx-4" href="#/leaderboard">
              Full Leaderboard
            </a>
          </div>
        </div>
        <div className="flex-1">
          <MiniLeaderboard />
        </div>
      </div>
    </div>
  );
}
