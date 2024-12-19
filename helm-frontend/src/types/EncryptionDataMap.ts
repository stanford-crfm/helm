export default interface EncryptionDetails {
  ciphertext: string;
  key: string;
  iv: string;
  tag: string;
}

export type EncryptionDataMap = Record<string, EncryptionDetails>;
