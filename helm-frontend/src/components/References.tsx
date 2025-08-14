import type Reference from "@/types/Reference";
import { Badge } from "@tremor/react";

interface Props {
  references: Reference[];
}

const CORRECT_TAG = "correct";

export default function References({ references }: Props) {
  return (
    <span>
      <h3>References</h3>
      <ul>
        {references.map((reference, index) => {
          return (
            <li
              key={index}
              className="bg-base-200 p-2 block overflow-auto w-full max-h-72 mb-2 whitespace-pre-wrap localize-text-direction"
            >
              {reference.output.text}
              {reference.tags.map((tag) => {
                return (
                  <Badge
                    className="mx-2"
                    color={tag === CORRECT_TAG ? "green" : undefined}
                  >
                    {tag}
                  </Badge>
                );
              })}
            </li>
          );
        })}
      </ul>
    </span>
  );
}
