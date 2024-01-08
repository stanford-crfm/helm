export default function getBenchmarkEndpoint(path: string): string {
  return `${window.BENCHMARK_OUTPUT_BASE_URL.replace(/\/$/, "")}/${path.replace(
    /^\//,
    "",
  )}`;
}
