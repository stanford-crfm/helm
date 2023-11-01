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
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
  const legacyRelease = (window as any).LEGACY_RELEASE;

  if (value.value === undefined) {
    return "-";
  }

  if (value.href !== undefined && ignoreHref === false) {
    const href = (() => {
      const matches = value.href.match(/group=([^&]+)/);
      if (matches === null) {
        return value.href;
      }

      return "/" + legacyRelease + `/groups/${matches[1]}`;
    })();
    return <Link to={href}>{formatNumber(value.value)}</Link>;
  }

  if (value.markdown) {
    return <MarkdownValue value={String(value.value)} />;
  }

  return <>{formatNumber(value.value)}</>;
}
