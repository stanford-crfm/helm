import MiniLeaderboard from "@/components/MiniLeaderboard";
import { Link } from "react-router-dom";

export default function SQLLanding() {
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl my-8 font-bold text-center">HELM SQL</h1>
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1 text-l">
          <p className="mb-4">
            Text-to-SQL is the task of converting natural language instructions
            into SQL code. There has been increasing interest in text-to-SQL for
            applications by data scientists in various domains. Thus, we
            introduce the <strong className="font-bold">HELM SQL</strong>{" "}
            leaderboard for text-to-SQL evaluations. The HELM SQL leaderboard
            evaluates leading LLMs on two existing text-to-SQL benchmarks
            (Spider, BIRD-SQL) that cover a range of professional domains. In
            addition, we introduce a new benchmark,{" "}
            <strong className="font-bold">CzechBankQA</strong>, a text-to-SQL
            benchmark based on a real public bank customer relational database,
            to address the lack of coverage of text-to-SQL in the financial
            domain. CzechBankQA consists of text-to-SQL queries and gold labels
            provided by professionals at Wells Fargo. We hope that this
            leaderboard provides useful insights for data science practitioners.
          </p>
          <p className="my-4">
            This leaderboard was produced through research collaboration with{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://www.wellsfargo.com/"
            >
              Wells Fargo
            </a>
            , and was funded by the{" "}
            <a
              className="font-bold underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
              href="https://hai.stanford.edu/corporate-affiliate-program"
            >
              HAI Corporate Affiliate Program
            </a>
            .
          </p>
          <div className="flex flex-row justify-center mt-4">
            <a className="px-10 btn rounded-md mx-4 hidden" href="#">
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
