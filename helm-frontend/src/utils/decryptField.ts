import { EncryptionDataMap } from "@/types/EncryptionDataMap";

async function decryptText(
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

export default async function decryptField(
  encryptedFieldValue: string,
  encryptionData: EncryptionDataMap,
): Promise<string> {
  if (!encryptedFieldValue.includes("encrypted_text")) {
    return encryptedFieldValue;
  }
  const encryptionDetails = encryptionData[encryptedFieldValue];
  if (encryptionDetails === undefined) {
    console.error(`Could not find ${encryptedFieldValue} in encryption data`);
    return encryptedFieldValue;
  }
  return await decryptText(
    encryptionDetails.ciphertext,
    encryptionDetails.key,
    encryptionDetails.iv,
    encryptionDetails.tag,
  );
}
