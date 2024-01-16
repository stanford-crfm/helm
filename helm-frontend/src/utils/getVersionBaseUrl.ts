export default function getVersionBaseUrl(): string {
  if (window.RELEASE) {
    return `/releases/${window.RELEASE}`;
  } else {
    return `/runs/${window.SUITE}`;
  }
}
