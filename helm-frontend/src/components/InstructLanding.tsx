import instructFlowchart from "@/assets/instruct/instruct-flowchart.svg";
import instructGraph from "@/assets/instruct/instruct-graph.svg";

export default function InstructLanding() {
  // TODO: Link to blog post.
  return (
    <div className="container mx-auto px-16">
      <h1 className="text-3xl mt-16 font-bold text-center">
        HELM Instruct: A Multidimensional Instruction Following Evaluation
        Framework with Absolute Ratings
      </h1>
      <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-8 md:gap-32 my-8">
        <a
          className="px-10 btn rounded-md"
          href="https://crfm.stanford.edu/2024/02/18/helm-instruct.html"
        >
          Blog Post
        </a>
        <a
          className="px-10 btn rounded-md"
          href="https://github.com/stanford-crfm/helm"
        >
          Github
        </a>
      </div>
      <p>
        We introduce <em>HELM Instruct</em>, a multidimensional evaluation
        framework for instruction-following LLMs with absolute ratings. The
        framework takes an instruction, a model, an evaluator, and a criterion
        to generate a score. In our study, we use HELM Instruct to compare 4
        instruction-following models on 7 scenarios based on 4 Human/LM
        evaluators and 5 criteria. Check out the blog post for more details.
      </p>
      <div className="grid my-16 grid-cols-1 md:mx-32 md:grid-cols-2 md:gap-2">
        <img
          src={instructFlowchart}
          alt="Evaluation flowchart"
          className="mx-auto block"
          sizes="100vw"
        />
        <img
          src={instructGraph}
          alt="7 scenarios, 4 models, 4 evaluators and 5 criteria"
          className="mx-auto block"
          sizes="100vw"
        />
      </div>
      <table className="rounded-lg shadow-md table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Average</th>
            <th>Helpfulness</th>
            <th>Understandability</th>
            <th>Completeness</th>
            <th>Conciseness</th>
            <th>Harmlessness</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>openai_gpt-4-0314</td>
            <td>4.63</td>
            <td>4.42</td>
            <td>4.85</td>
            <td>4.50</td>
            <td>4.42</td>
            <td>4.95</td>
          </tr>
          <tr>
            <td>openai_gpt-3.5-turbo-0613</td>
            <td>4.60</td>
            <td>4.34</td>
            <td>4.86</td>
            <td>4.42</td>
            <td>4.41</td>
            <td>4.97</td>
          </tr>
          <tr>
            <td>anthropic_claude-v1.3</td>
            <td>4.56</td>
            <td>4.25</td>
            <td>4.87</td>
            <td>4.32</td>
            <td>4.40</td>
            <td>4.97</td>
          </tr>
          <tr>
            <td>cohere_command-xlarge-beta</td>
            <td>4.31</td>
            <td>3.90</td>
            <td>4.73</td>
            <td>3.88</td>
            <td>4.31</td>
            <td>4.72</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
