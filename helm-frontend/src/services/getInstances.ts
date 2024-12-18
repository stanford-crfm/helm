import Instance from "@/types/Instance";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";
import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

// Helper function for decryption
async function decryptField(ciphertext: string, key: string, iv: string, tag: string): Promise<string> {
  // Convert Base64 strings to Uint8Array
  const decodeBase64 = (str: string) =>
    Uint8Array.from(atob(str), c => c.charCodeAt(0));

  const cryptoKey = await window.crypto.subtle.importKey(
    "raw",
    decodeBase64(key),
    "AES-GCM",
    true,
    ["decrypt"]
  );

  const combinedCiphertext = new Uint8Array([
    ...decodeBase64(ciphertext),
    ...decodeBase64(tag),
  ]);

  const ivArray = decodeBase64(iv);

  const decrypted = await window.crypto.subtle.decrypt(
    { name: "AES-GCM", iv: ivArray },
    cryptoKey,
    combinedCiphertext
  );

  return new TextDecoder().decode(decrypted);
}

// Main function to fetch and decrypt instances
export default async function getInstancesByRunName(
  runName: string,
  signal: AbortSignal,
  suite?: string,
  userAgreed?: boolean,
): Promise<Instance[]> {
  try {
    // Fetch instances JSON
    const response = await fetch(
      getBenchmarkEndpoint(
        `/runs/${suite || getBenchmarkSuite()}/${runName}/instances.json`,
      ),
      { signal },
    );
    const instances = (await response.json()) as Instance[];

    // Decrypt text fields if runName contains "gpqa"
    if (runName.includes("gpqa") && userAgreed) {
      const encryptionResponse = await fetch(
        getBenchmarkEndpoint(
          `/runs/${suite || getBenchmarkSuite()}/${runName}/encryption_data.json`,
        ),
        { signal },
      );
      const encryptionData = await encryptionResponse.json();

      for (const instance of instances) {
        const inputEncryption = encryptionData[instance.input.text];
        if (inputEncryption) {
          instance.input.text = "redacted";
          instance.input.text = await decryptField(
            inputEncryption.ciphertext,
            inputEncryption.key,
            inputEncryption.iv,
            inputEncryption.tag
          );
        }
  
        for (const reference of instance.references) {
          const referenceEncryption = encryptionData[reference.output.text];
          if (referenceEncryption) {
            reference.output.text = await decryptField(
              referenceEncryption.ciphertext,
              referenceEncryption.key,
              referenceEncryption.iv,
              referenceEncryption.tag
            );
          }
        }
      }
    }

    // print first 10 instances
    for (let i = 0; i < 10; i++) {
      console.log(instances[i]);
    }

    return instances;
  } catch (error) {
    if (error instanceof Error && error.name !== "AbortError") {
      console.log(error);
    }
    return [];
  }
}
