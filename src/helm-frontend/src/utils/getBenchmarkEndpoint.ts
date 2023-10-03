export default function getBenchmarkEndpoint(path: string): string {
  return `${import.meta.env.VITE_HELM_BENCHMARKS_ENDPOINT.replace(/\/$/, "")}/${
    path.replace(/^\//, "")
  }`;
}
