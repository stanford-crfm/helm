import type { ReactNode } from "react";

interface Props {
  active?: boolean;
  onClick: () => void;
  children: ReactNode;
  size?: "sm" | "md" | "lg" | "xl";
}

export default function Tab(
  { active = false, onClick = (() => {}), size = "md", children }: Props,
) {
  return (
    <div
      onClick={onClick}
      className={`whitespace-nowrap text-${size} mb-[-2px] text-md tab tab-bordered${
        active ? " tab-active" : " border-none"
      }`}
    >
      {children}
    </div>
  );
}
