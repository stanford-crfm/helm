import type DisplayRequest from "@/types/DisplayRequest";
import { EncryptionDataMap } from "@/types/EncryptionDataMap";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import isScenarioEncrypted from "@/utils/isScenarioEncrypted";

// Helper function for decryption
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

export default async function getDisplayRequestsByName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
  userAgreed?: boolean,
): Promise<DisplayRequest[]> {
  try {
    const response = await fetch(
      getBenchmarkEndpoint(
        `/runs/${
          suite || getBenchmarkSuite()
        }/${runName}/display_requests.json`,
      ),
      { signal },
    );
    const displayRequests = (await response.json()) as DisplayRequest[];

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

      for (const request of displayRequests) {
        const encryptedPrompt = request.request.prompt;
        const encryptionDetails = encryptionData[encryptedPrompt];

        if (encryptionDetails) {
          try {
            request.request.prompt = await decryptField(
              encryptionDetails.ciphertext,
              encryptionDetails.key,
              encryptionDetails.iv,
              encryptionDetails.tag,
            );
          } catch (error) {
            console.error(
              `Failed to decrypt prompt for instance_id: ${request.instance_id}`,
              error,
            );
          }
        }
      }
    }

    return displayRequests;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
