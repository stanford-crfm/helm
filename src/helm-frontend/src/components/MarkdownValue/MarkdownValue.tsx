import type { PropsWithChildren } from "react";
import ReactMarkdown from "react-markdown";

interface LinkProps {
  href?: string;
}

interface Props {
  value: string;
}

function Link({ href, children }: PropsWithChildren<LinkProps>) {
  return (
    <a
      href={href}
      className="link link-primary link-hover"
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  );
}

/**
 * We add a "link" class to all links in the markdown
 * for styling purposes in daisyui
 */
export default function MarkdownValue({ value }: Props) {
  return (
    <span>
      <ReactMarkdown
        components={{
          a: Link,
        }}
        children={value}
      />
    </span>
  );
}
