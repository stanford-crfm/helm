export default function isScenarioEncrypted(runName: string): boolean {
  return runName.includes("gpqa") || runName.includes("ewok");
}
