import type DisplayPrediction from "@/types/DisplayPrediction";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default async function getDisplayPredictionsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
): Promise<DisplayPrediction[]> {
  try {
    const displayPrediction = await fetch(
      getBenchmarkEndpoint(
        `/benchmark_output/runs/${
          suite || getBenchmarkSuite()
        }/${runName}/display_predictions.json`,
      ),
      { signal },
    );

    return (await displayPrediction.json()) as DisplayPrediction[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
