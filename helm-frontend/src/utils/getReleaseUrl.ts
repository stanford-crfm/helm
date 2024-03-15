import ReleaseIndexEntry from "@/types/ReleaseIndexEntry";

export default function getReleaseUrl(
  version: string | undefined,
  currReleaseIndexEntry: ReleaseIndexEntry | undefined,
): string {
  if (!currReleaseIndexEntry) {
    return "#";
  }
  if (!version) {
    return `https://crfm.stanford.edu/helm/${currReleaseIndexEntry.id}/`;
  }
  return `https://crfm.stanford.edu/helm/${currReleaseIndexEntry.id}/`;
}
