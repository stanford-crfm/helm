import type RunGroup from "@/types/RunGroup";

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
    <ul>
      <h3 className="text-3xl">{runGroups.length} Scenarios</h3>
      {topLevelGroups.filter((topLevelGroup) =>
        subGroups.some((subGroup) =>
          (topLevelGroup.subgroups || []).includes(subGroup.name)
        )
      ).map((topLevelGroup, idx) => (
        <li key={idx}>
          <ul>
            <h2>{topLevelGroup.display_name}</h2>
            {subGroups.filter((subGroup) =>
              (topLevelGroup.subgroups || []).includes(subGroup.name)
            ).map((subGroup, idx) => (
              subGroup.todo
                ? (
                  <li key={idx} className="ml-4 text-slate-300">
                    {subGroup.display_name}
                  </li>
                )
                : (
                  <li key={idx} className="ml-4">
                    {subGroup.display_name}
                  </li>
                )
            ))}
          </ul>
        </li>
      ))}
    </ul>
  );
}
