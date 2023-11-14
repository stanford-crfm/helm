export default function getVersionBaseUrl(): string {
  if (window.RELEASE) {
    return `/benchmark_output/releases/${window.RELEASE}`;
  } else {
    return `/benchmark_output/runs/${window.SUITE}`;
  }
}
