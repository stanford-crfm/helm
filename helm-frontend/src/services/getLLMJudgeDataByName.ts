import type LLMJudgeData from "@/types/LLMJudgeData";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";


export default async function getLLMJudgeDataByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<LLMJudgeData | undefined> {
  const fileName = "llm_judge_summary.json";

  try {
    const response = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${runName}/${fileName}`,
      ),
      { signal },
    );

    if (!response.ok) {
      if (response.status !== 404) {
        console.warn(
          `LLM Judge summary: Request failed for run '${runName}' with status ${response.status}. File: ${fileName}`,
        );
      }
      return undefined;
    }

    const data = await response.json();
    return data as LLMJudgeData;

  } catch (error) {
    if (error instanceof Error) {
      if (error.name !== "AbortError") {
        console.warn(
          `LLM Judge summary: Error fetching or parsing data for run '${runName}'. File: ${fileName}. Error: ${error.message}`,
        );
      }
    } else {
      console.warn(
        `LLM Judge summary: An unknown error occurred while fetching data for run '${runName}'. File: ${fileName}.`,
        error,
      );
    }
    return undefined;
  }
}