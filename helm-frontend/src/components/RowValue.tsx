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

export default function RowValue({ value }: Props) {
  if (value.value === undefined) {
    return "-";
  }

  if (value.run_spec_names) {
    const href = (() => {
      if (value.run_spec_names.length == 1) {
        return "/runs/" + value.run_spec_names[0];
      } else if (value.run_spec_names.length > 1) {
        const rawHref =
          "/runs/?q=" +
          value.run_spec_names.map((name) => `^${name}$`).join("|");
        const href = encodeURI(rawHref);
        return href;
      }
    })();
    if (href) {
      return (
        <Link to={href} inTable>
          {formatNumber(value.value)}
        </Link>
      );
    } else {
      return <>{formatNumber(value.value)}</>;
    }
  }

  if (value.markdown) {
    return <MarkdownValue value={String(value.value)} />;
  }

  return <>{formatNumber(value.value)}</>;
}
