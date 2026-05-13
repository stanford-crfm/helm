import type DisplayRequest from "@/types/DisplayRequest";
import { EncryptionDataMap } from "@/types/EncryptionDataMap";
import decryptField from "@/utils/decryptField";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import isScenarioEncrypted from "@/utils/isScenarioEncrypted";

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
        request.request.prompt = await decryptField(
          request.request.prompt,
          encryptionData,
        );
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
