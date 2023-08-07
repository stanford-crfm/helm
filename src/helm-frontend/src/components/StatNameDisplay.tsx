import type Stat from "@/types/Stat";
import underscoreToTitle from "@/utils/underscoreToTitle";

interface Props {
  stat: Stat;
}

export default function StatNameDisplay({ stat }: Props) {
  const value = `${
    stat.name.split !== undefined ? ` on ${stat.name.split}` : ""
  }${stat.name.sub_split !== undefined ? `/${stat.name.sub_split}` : ""}${
    stat.name.perturbation !== undefined
      ? ` with ${stat.name.perturbation.name}`
      : " original"
  }`;
  return (
    <span>
      <strong>{underscoreToTitle(stat.name.name)}</strong>
      {value}
    </span>
  );
}
