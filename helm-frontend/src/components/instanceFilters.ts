import type Instance from "@/types/Instance";

function normalize(value: string): string {
  return value.trim().toLowerCase();
}

function getSearchableText(instance: Instance): string {
  const referenceText = instance.references
    .map((reference) => reference.output.text)
    .join(" ");

  return normalize(
    [
      instance.id,
      instance.split,
      instance.perturbation?.name ?? "",
      instance.input.text,
      referenceText,
    ].join(" "),
  );
}

export function matchesInstanceSearch(
  instance: Instance,
  query: string,
): boolean {
  const normalizedQuery = normalize(query);

  if (!normalizedQuery) {
    return true;
  }

  return getSearchableText(instance).includes(normalizedQuery);
}
