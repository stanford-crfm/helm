import ReleaseIndexEntry from "@/types/ReleaseIndexEntry";

export default function getReleaseUrl(
  version: string,
  currReleaseIndexEntry: ReleaseIndexEntry | undefined,
): string {
  return currReleaseIndexEntry !== undefined &&
    currReleaseIndexEntry.id !== undefined
    ? `https://crfm.stanford.edu/helm/${currReleaseIndexEntry.id}/${version}/`
    : `https://crfm.stanford.edu/helm/lite/${version}/`;
}
