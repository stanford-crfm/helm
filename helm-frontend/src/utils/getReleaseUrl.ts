export default function getReleaseUrl(
  version: string | undefined,
  currProjectId: string | undefined,
): string {
  if (!currProjectId) {
    return "#";
  }
  if (currProjectId === "home") {
    return `https://crfm.stanford.edu/helm/`;
  }
  if (!version) {
    return `https://crfm.stanford.edu/helm/${currProjectId}/latest/`;
  }
  return `https://crfm.stanford.edu/helm/${currProjectId}/${version}/`;
}
