// hardcoded function to find the first stat that we consider a "main" stat
// returns its name and whether it should be perceived as correct

export default function getStatCorrectness(
  stats: Record<string, number>,
): [string, boolean] {
  // the order of this implicitly defines priority of which we consider to be a main metric
  const threshold = 0.5;

  const lowerIsBetterMap: Record<string, boolean> = {
    quasi_exact_match: false,
    toxic_frac: true,
    safety_score: false,
    exact_match: false,
  };
  const statKeys = Object.keys(stats);

  for (const statKey of statKeys) {
    if (stats[statKey] !== undefined) {
      if (lowerIsBetterMap[statKey]) {
        if (stats[statKey] < threshold) {
          return [statKey, true];
        }
        return [statKey, false];
      }
      if (stats[statKey] >= threshold) {
        return [statKey, true];
      }
      return [statKey, false];
    }
  }

  return ["", false];
}
