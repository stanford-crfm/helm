import Preview from "./Preview";

// TODO: This is a dirty hack to support extra data from WildBench
// but eventually we should make sure all extra data are supported generally.
type Props = {
  extraData: Record<
    string,
    string | number | null | Array<string | number | null>
  >;
};

function valueDisplay(value: string | number | null) {
  return (
    <Preview
      value={
        value === null
          ? "null"
          : typeof value === "object"
          ? JSON.stringify(value)
          : value.toString()
      }
    />
  );
}

function listExtraDataDisplay(listData: Array<string | number | null>) {
  // TODO: Elements inside the map need keys
  return <div>{listData.map((value) => valueDisplay(value))}</div>;
}

export default function ExtraDataDisplay({ extraData }: Props) {
  return (
    <details className="collapse collapse-arrow border rounded-md bg-white my-2">
      <summary className="collapse-title">View instance extra data</summary>

      <div className="collapse-content">
        {Object.entries(extraData).map(([key, value]) => (
          <div>
            <h3 className="ml-1">{key}</h3>
            {Array.isArray(value)
              ? listExtraDataDisplay(value)
              : valueDisplay(value)}
          </div>
        ))}
      </div>
    </details>
  );
}
