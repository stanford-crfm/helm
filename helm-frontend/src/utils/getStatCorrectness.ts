// hardcoded function to find the first stat that we consider a "main" stat
// returns its name and whether it should be perceived as correct

export default function getStatCorrectness(
  stats: Record<string, number>,
): [string, boolean] {
  // the order of this implicitly defines priority of which we consider to be a main metric
  const statKeys = [
    "quasi_exact_match",
    "toxic_frac",
    "safety_score",
    "exact_match",
  ];

  for (const statKey of statKeys) {
    if (stats[statKey] !== undefined) {
      if (stats[statKey] > 0) {
        return [statKey, true];
      }
      return [statKey, false];
    }
  }

  return ["", false];
}
