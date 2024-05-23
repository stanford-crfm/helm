import MarkdownValue from "@/components/MarkdownValue";
import type RowValueType from "@/types/RowValue";
import Link from "@/components/Link";
import { ArrowTopRightOnSquareIcon } from "@heroicons/react/20/solid";

interface Props {
  value: RowValueType;
  title?: string;
  hideIcon?: boolean;
  ignoreHref?: boolean;
}

function formatNumber(value: string | number): string {
  if (Number.isNaN(Number(value))) {
    return String(value);
  }

  return String(Math.round(Number(value) * 1000) / 1000);
}

export default function RowValue({ value, title, hideIcon }: Props) {
  // TODO remove this once we stop adding ⚠ to output JSONs
  if (typeof value.value === "string" && value.value.includes("⚠")) {
    value.value = value.value.replace("⚠", "");
  }

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
        <Link to={href} inTable title={title}>
          <div className="flex items-center ">
            {formatNumber(value.value)}
            {!hideIcon && (
              <ArrowTopRightOnSquareIcon
                className="w-3 h-3 ml-1"
                stroke="#cbcbcb"
                fill="#cbcbcb"
              />
            )}
          </div>
        </Link>
      );
    } else {
      if (title) {
        return <a title={title}>{formatNumber(value.value)}</a>;
      } else {
        return <>{formatNumber(value.value)}</>;
      }
    }
  }

  if (value.href) {
    return (
      <Link to={value.href} inTable title={title}>
        <div className="flex items-center">
          {formatNumber(value.value)}
          {!hideIcon && (
            <ArrowTopRightOnSquareIcon
              className="w-3 h-3 ml-1"
              stroke="#cbcbcb"
              fill="#cbcbcb"
            />
          )}
        </div>
      </Link>
    );
  }

  if (value.markdown) {
    return <MarkdownValue value={String(value.value)} />;
  }

  if (title) {
    return <a title={title}>{formatNumber(value.value)}</a>;
  }
  return <>{formatNumber(value.value)}</>;
}
