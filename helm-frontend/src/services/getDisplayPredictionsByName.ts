import type DisplayPrediction from "@/types/DisplayPrediction";
import { EncryptionDataMap } from "@/types/EncryptionDataMap";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import isScenarioEncrypted from "@/utils/isScenarioEncrypted";

async function decryptField(
  ciphertext: string,
  key: string,
  iv: string,
  tag: string,
): Promise<string> {
  const decodeBase64 = (str: string) =>
    Uint8Array.from(atob(str), (c) => c.charCodeAt(0));

  const cryptoKey = await window.crypto.subtle.importKey(
    "raw",
    decodeBase64(key),
    "AES-GCM",
    true,
    ["decrypt"],
  );

  const combinedCiphertext = new Uint8Array([
    ...decodeBase64(ciphertext),
    ...decodeBase64(tag),
  ]);

  const ivArray = decodeBase64(iv);

  const decrypted = await window.crypto.subtle.decrypt(
    { name: "AES-GCM", iv: ivArray },
    cryptoKey,
    combinedCiphertext,
  );

  return new TextDecoder().decode(decrypted);
}

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
        const encryptedText = prediction.predicted_text;
        const encryptionDetails = encryptionData[encryptedText];

        if (encryptionDetails) {
          try {
            prediction.predicted_text = await decryptField(
              encryptionDetails.ciphertext,
              encryptionDetails.key,
              encryptionDetails.iv,
              encryptionDetails.tag,
            );
          } catch (error) {
            console.error(
              `Failed to decrypt predicted_text for instance_id: ${prediction.instance_id}`,
              error,
            );
          }
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
