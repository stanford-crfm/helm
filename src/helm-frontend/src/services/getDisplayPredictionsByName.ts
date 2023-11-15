import type DisplayPrediction from "@/types/DisplayPrediction";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getVersionBaseUrl from "@/utils/getVersionBaseUrl";

export default async function getDisplayPredictionsByName(
  runName: string,
  signal: AbortSignal,
): Promise<DisplayPrediction[]> {
  try {
    const displayPrediction = await fetch(
      getBenchmarkEndpoint(
        `${getVersionBaseUrl()}/${runName}/display_predictions.json`,
      ),
      { signal },
    );

    return (await displayPrediction.json()) as DisplayPrediction[];
  } catch (error) {
    console.log(error);
    return [];
  }
}
