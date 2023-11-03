import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function Tabs({ children }: Props) {
  return (
    <div
      role="navigation"
      className="tabs flex-nowrap border-b-2 border-gray-2 overflow-x-auto overflow-y-hidden"
    >
      {children}
    </div>
  );
}
