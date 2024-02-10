import { List, ListItem } from "@tremor/react";
import type DisplayRequest from "@/types/DisplayRequest";
import Preview from "../Preview";

type Props = {
  request: DisplayRequest;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function formatArray(arr: any): any {
  if (!Array.isArray(arr)) {
    return arr;
  }

  if (arr.length === 0) {
    return "";
  }

  return String(
    `[${arr.map((x) => String(x).replace(/\n/, "\\n")).join(", ")}]`,
  );
}

export default function Request({ request }: Props) {
  return (
    <div>
      <h3 className="block text text-gray-400">Prompt</h3>
      <Preview value={request.request.prompt} />

      <List>
        {(Object.keys(request.request) as (keyof typeof request.request)[])
          .filter((key) => key !== "prompt")
          .map((key, idx) => (
            <ListItem key={idx + 1}>
              <span>{key}:</span>
              <span>{formatArray(request.request[key])}</span>
            </ListItem>
          ))}
      </List>
    </div>
  );
}
