import Instance from "@/types/Instance";
import { EncryptionDataMap } from "@/types/EncryptionDataMap";
import decryptField from "@/utils/decryptField";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";
import isScenarioEncrypted from "@/utils/isScenarioEncrypted";

export default async function getInstancesByRunName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
  userAgreed?: boolean,
): Promise<Instance[]> {
  try {
    const response = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${runName}/instances.json`,
      ),
      { signal },
    );
    const instances = (await response.json()) as Instance[];

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

      for (const instance of instances) {
        instance.input.text = await decryptField(
          instance.input.text,
          encryptionData,
        );

        for (const reference of instance.references) {
          reference.output.text = await decryptField(
            reference.output.text,
            encryptionData,
          );
        }
      }
    }

    return instances;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
