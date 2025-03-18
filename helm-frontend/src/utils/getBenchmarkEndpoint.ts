export default function getBenchmarkEndpoint(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${window.BENCHMARK_OUTPUT_BASE_URL.replace(/\/$/, "")}/${path
    .replace(/^\//, "")
    .split("/")
    .map((component) => encodeURIComponent(component))
    .join("/")}`;
}
