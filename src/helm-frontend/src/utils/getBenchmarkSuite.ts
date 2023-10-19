export default function getBenchmarkSuite(): string {
  return String(import.meta.env.VITE_HELM_BENCHMARKS_SUITE);
}
