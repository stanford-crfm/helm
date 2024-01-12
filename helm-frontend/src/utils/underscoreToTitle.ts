export default function underscoreToTitle(value: string) {
  return String(value)
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}
