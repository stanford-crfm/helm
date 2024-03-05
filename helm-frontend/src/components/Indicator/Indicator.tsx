import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/24/outline";
import { Badge } from "@tremor/react";

interface BadgeProps {
  value: string | number;
}

function BadgeSuccess({ value }: BadgeProps) {
  return (
    <Badge icon={CheckCircleIcon} color="green">
      {value}
    </Badge>
  );
}

function BadgeError({ value }: BadgeProps) {
  return (
    <Badge icon={XCircleIcon} color="red">
      {value}
    </Badge>
  );
}

interface Props {
  stats: {
    [key: string]: number;
  };
}

export default function Indicator({ stats }: Props) {
  if (stats["quasi_exact_match"] !== undefined) {
    if (stats["quasi_exact_match"]) {
      return (
        <BadgeSuccess
          value={`quasi exact match: ${stats["quasi_exact_match"]}`}
        />
      );
    }
    return (
      <BadgeError value={`quasi exact match: ${stats["quasi_exact_match"]}`} />
    );
  }

  if (stats["toxic_frac"] !== undefined) {
    if (stats["toxic_frac"] > 0) {
      return <BadgeError value={`toxic frac: ${stats["toxic_frac"]}`} />;
    }
    return <BadgeSuccess value={`toxic frac: ${stats["toxic_frac"]}`} />;
  }

  /**
   * @TODO is this always 1 or 0
   */
  if (stats["exact_match"] !== undefined) {
    if (stats["exact_match"] > 0) {
      return <BadgeSuccess value={`exact match: ${stats["exact_match"]}`} />;
    }
    return <BadgeError value={`exact match: ${stats["exact_match"]}`} />;
  }
  
  if (stats["accuracy"] !== undefined) {
    if (stats["accuracy"] > 0) {
      return <BadgeSuccess value={`accuracy: ${stats["accuracy"]}`} />;
    }
    return <BadgeError value={`accuracy: ${stats["accuracy"]}`} />;
  }
}
