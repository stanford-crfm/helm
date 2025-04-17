import { List, ListItem } from "@tremor/react";
import type DisplayRequest from "@/types/DisplayRequest";
import Preview from "../Preview";
import MessagesDisplay from "../MessagesDisplay";
import MultimediaObjectDisplay from "../MultimediaObjectDisplay";

type Props = {
  request: DisplayRequest;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function formatArray(arr: any): any {
  if (!Array.isArray(arr)) {
    return String(arr);
  }

  if (arr.length == 0) {
    return "[]";
  }

  return String(
    `[${arr.map((x) => String(x).replace(/\n/, "\\n")).join(", ")}]`,
  );
}

export default function Request({ request }: Props) {
  return (
    <div>
      {request.request.prompt.length > 0 ? (
        <div>
          <h3 className="block text text-gray-400">
            Prompt ({request.request.prompt.length} Chars)
          </h3>
          <Preview value={request.request.prompt} />
        </div>
      ) : request.request.multimodal_prompt ? (
        <div>
          <h3 className="block text text-gray-400">Prompt</h3>
          <MultimediaObjectDisplay
            multimediaObject={request.request.multimodal_prompt}
          />
        </div>
      ) : request.request.messages && request.request.messages.length ? (
        <div>
          <h3 className="block text text-gray-400">Prompt</h3>
          <MessagesDisplay messages={request.request.messages} />
        </div>
      ) : (
        <h3 className="block text text-gray-400">Empty Prompt</h3>
      )}
      <List>
        {(Object.keys(request.request) as (keyof typeof request.request)[])
          .filter((key) => key !== "prompt")
          .map((key, idx) => (
            <ListItem key={idx + 1}>
              <span>{key}:</span>
              {request.request && request.request[key] ? (
                <span>{formatArray(request.request[key])}</span>
              ) : (
                "null"
              )}
            </ListItem>
          ))}
      </List>
    </div>
  );
}
