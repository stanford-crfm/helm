import ProjectMetadata from "@/types/ProjectMetadata";

export default function getReleaseUrl(
  version: string | undefined,
  currProjectMetadata: ProjectMetadata | undefined,
): string {
  if (!currProjectMetadata) {
    return "#";
  }
  if (!version) {
    return `https://crfm.stanford.edu/helm/${currProjectMetadata.id}/latest/`;
  }
  return `https://crfm.stanford.edu/helm/${currProjectMetadata.id}/${version}/`;
}
