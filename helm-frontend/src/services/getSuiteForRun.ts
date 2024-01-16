import getBenchmarkRelease from "@/utils/getBenchmarkRelease";

export default function getSuiteForRun(
  runNameToSuite: Record<string, string>,
  runName: string,
) {
  const suite = getBenchmarkRelease() ? runNameToSuite[runName] : window.SUITE;
  return suite;
}
