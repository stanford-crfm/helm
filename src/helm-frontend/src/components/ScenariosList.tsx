import type RunGroup from "@/types/RunGroup";
import { Link as ReactRouterLink } from "react-router-dom";

interface Props {
  runGroups: RunGroup[];
}

export default function ScenariosList({ runGroups }: Props) {
  const { topLevelGroups, subGroups } = runGroups.reduce(
    (acc, cur) => {
      if (cur.category !== undefined) {
        acc.topLevelGroups.push(cur);
      } else {
        acc.subGroups.push(cur);
      }

      return acc;
    },
    { topLevelGroups: [], subGroups: [] } as {
      topLevelGroups: RunGroup[];
      subGroups: RunGroup[];
    },
  );

  return (
    <section>
      <h3 className="text-3xl">{runGroups.length} scenarios</h3>
      <ul>
        {topLevelGroups
          .filter((topLevelGroup) =>
            subGroups.some((subGroup) =>
              (topLevelGroup.subgroups || []).includes(subGroup.name),
            ),
          )
          .map((topLevelGroup, idx) => (
            <li key={idx} className="my-3">
              <ReactRouterLink
                className="text-black"
                to={"groups/" + topLevelGroup.name}
              >
                <h2>{topLevelGroup.display_name}</h2>
              </ReactRouterLink>
              <ul className="list-disc list-inside">
                {subGroups
                  .filter((subGroup) =>
                    (topLevelGroup.subgroups || []).includes(subGroup.name),
                  )
                  .map((subGroup, idx) =>
                    subGroup.todo || subGroup.name.includes("CLEVA") ? (
                      <li
                        key={idx}
                        className={`${
                          subGroup.todo ? "ml-4 text-slate-300" : "ml-4"
                        }`}
                      >
                        {subGroup.display_name}
                      </li>
                    ) : (
                      <ReactRouterLink
                        className="text-black"
                        to={"groups/" + subGroup.name}
                      >
                        <li
                          key={idx}
                          className={`${
                            subGroup.todo ||
                            subGroup.display_name.includes("CLEVA")
                              ? "ml-4 text-slate-300"
                              : "ml-4"
                          }`}
                        >
                          {subGroup.display_name}
                        </li>
                      </ReactRouterLink>
                    ),
                  )}
              </ul>
            </li>
          ))}
      </ul>
    </section>
  );
}
