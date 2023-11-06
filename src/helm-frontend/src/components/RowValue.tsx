import MarkdownValue from "@/components/MarkdownValue";
import type RowValueType from "@/types/RowValue";
import Link from "@/components/Link";

interface Props {
  value: RowValueType;
  ignoreHref?: boolean;
}

function formatNumber(value: string | number): string {
  if (Number.isNaN(Number(value))) {
    return String(value);
  }

  return String(Math.round(Number(value) * 1000) / 1000);
}

export default function RowValue({ value, ignoreHref = false }: Props) {
  if (value.value === undefined) {
    return "-";
  }

  if (value.href !== undefined && ignoreHref === false) {
    const href = (() => {
      const matches = value.href.match(/group=([^&]+)/);
      if (matches === null) {
        return value.href;
      }

      return `/groups/${matches[1]}`;
    })();
    return <Link to={href}>{formatNumber(value.value)}</Link>;
  }

  if (value.markdown) {
    return <MarkdownValue value={String(value.value)} />;
  }

  return <>{formatNumber(value.value)}</>;
}
