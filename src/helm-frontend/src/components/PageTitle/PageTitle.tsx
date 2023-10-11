import MarkdownValue from "../MarkdownValue";

interface Props {
  title: string;
  subtitle?: string;
  markdown?: boolean;
  className?: string;
}

export default function PageTitle({
  title,
  subtitle,
  markdown = false,
  className,
}: Props) {
  return (
    <header className={`m-8 ml-0 ${className}`}>
      <h1 className="text-4xl">{title}</h1>
      {markdown && subtitle !== undefined ? (
        <h2 className="mt-2 text-neutral">
          <MarkdownValue value={subtitle} />
        </h2>
      ) : (
        subtitle !== undefined && (
          <h2 className="mt-2 text-neutral">{subtitle}</h2>
        )
      )}
    </header>
  );
}
