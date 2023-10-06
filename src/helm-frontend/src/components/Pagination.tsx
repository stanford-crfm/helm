interface Props {
  currentPage: number;
  totalPages: number;
  onNextPage: () => void;
  onPrevPage: () => void;
  className?: string;
}

export default function Pagination({
  currentPage,
  totalPages,
  onNextPage,
  onPrevPage,
  className,
}: Props) {
  let mergedClassName = "join";
  if (className !== undefined) {
    mergedClassName = `join ${className}`;
  }

  return (
    <div className={mergedClassName}>
      <button onClick={onPrevPage} className="join-item btn">
        «
      </button>
      <button className="join-item btn">
        Page {currentPage} of {totalPages}
      </button>
      <button onClick={onNextPage} className="join-item btn">
        »
      </button>
    </div>
  );
}
