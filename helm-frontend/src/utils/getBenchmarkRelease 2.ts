export default function getBenchmarkRelease(): string | undefined {
  return window.RELEASE !== undefined ? window.RELEASE : undefined;
}
