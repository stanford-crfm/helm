export default function getBenchmarkSuite(): string | undefined {
  return window.SUITE !== undefined ? window.SUITE : undefined;
}
