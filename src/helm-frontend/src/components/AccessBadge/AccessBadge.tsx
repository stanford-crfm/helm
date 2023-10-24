import type { AccessLevel } from "@/types/Model";
import { Badge } from "@tremor/react";
import type { Color } from "@tremor/react";

type BadgeColorMap = {
  [key in AccessLevel]: Color;
};

type BadgeAccessMap = {
  [key in AccessLevel]: "Open" | "Limited" | "Closed";
};

const badgeColorMap: BadgeColorMap = {
  open: "green",
  limited: "yellow",
  closed: "red",
};

const badgeNameMap: BadgeAccessMap = {
  open: "Open",
  limited: "Limited",
  closed: "Closed",
};

interface Props {
  level: AccessLevel;
}

export default function AccessBadge({ level }: Props) {
  return <Badge color={badgeColorMap[level]}>{badgeNameMap[level]}</Badge>;
}
