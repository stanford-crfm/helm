import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/24/outline";
import { Badge } from "@tremor/react";
import getStatCorrectness from "@/utils/getStatCorrectness";

// Return a correctness indicator for the first matching stat
export default function Indicator(stats: { stats: Record<string, number> }) {
  const [statName, correctness] = getStatCorrectness(stats.stats);

  // iterate through stats.stats keys and return success
  if (statName === "") {
    return <></>;
  }

  return correctness ? (
    <BadgeSuccess value={`${statName}: ${stats.stats[statName]}`} />
  ) : (
    <BadgeError value={`${statName}: ${stats.stats[statName]}`} />
  );
}

function BadgeSuccess({ value }: { value: string }) {
  return (
    <Badge icon={CheckCircleIcon} color="green">
      {value}
    </Badge>
  );
}

function BadgeError({ value }: { value: string }) {
  return (
    <Badge icon={XCircleIcon} color="red">
      {value}
    </Badge>
  );
}
