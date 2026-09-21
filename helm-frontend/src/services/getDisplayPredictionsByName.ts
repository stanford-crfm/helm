import type DisplayPrediction from "@/types/DisplayPrediction";
import { EncryptionDataMap } from "@/types/EncryptionDataMap";
import decryptField from "@/utils/decryptField";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import isScenarioEncrypted from "@/utils/isScenarioEncrypted";

export default async function getDisplayPredictionsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
  userAgreed?: boolean,
): Promise<DisplayPrediction[]> {
  try {
    const response = await fetch(
      getBenchmarkEndpoint(
        `/runs/${
          suite || getBenchmarkSuite()
        }/${runName}/display_predictions.json`,
      ),
      { signal },
    );
    const displayPredictions = (await response.json()) as DisplayPrediction[];

    if (isScenarioEncrypted(runName) && userAgreed) {
      const encryptionResponse = await fetch(
        getBenchmarkEndpoint(
          `/runs/${
            suite || getBenchmarkSuite()
          }/${runName}/encryption_data.json`,
        ),
        { signal },
      );
      const encryptionData =
        (await encryptionResponse.json()) as EncryptionDataMap;

      for (const prediction of displayPredictions) {
        prediction.predicted_text = await decryptField(
          prediction.predicted_text,
          encryptionData,
        );

        if (prediction.thinking_text !== undefined) {
          prediction.thinking_text = await decryptField(
            prediction.thinking_text,
            encryptionData,
          );
        }
      }
    }

    return displayPredictions;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      console.log(error);
    }
    return [];
  }
}
