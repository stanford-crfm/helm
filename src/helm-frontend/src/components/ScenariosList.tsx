import type RunGroup from "@/types/RunGroup";
import { Link as ReactRouterLink } from "react-router-dom";

interface Props {
  runGroups: RunGroup[];
}

export default function ScenariosList({ runGroups }: Props) {
  // A run group is a scenario if it has metric groups but no subgroups.
  const scenariosByName = new Map<string, RunGroup>(
    runGroups
      .filter(
        (runGroup) =>
          runGroup.metric_groups !== undefined &&
          (runGroup.subgroups === undefined || runGroup.subgroups.length === 0),
      )
      .map((runGroup) => [runGroup.name, runGroup]),
  );

  // Only count scenarios that have a category and are displayed
  // i.e. don't count "orphaned" scenarios
  // Also, don't double-count scenarios that appear in multiple categories
  const categorizedScenarioNames = new Set<string>();

  const categoriesWithScenarios: [RunGroup, RunGroup[]][] = [];
  runGroups.forEach((runGroup) => {
    const subgroups: string[] = runGroup.subgroups ? runGroup.subgroups : [];
    const groupScenarios: RunGroup[] = [];
    subgroups.forEach((subgroup) => {
      const maybeScenario = scenariosByName.get(subgroup);
      if (maybeScenario) {
        groupScenarios.push(maybeScenario);
        categorizedScenarioNames.add(maybeScenario.name);
      }
    });
    if (groupScenarios.length > 0) {
      categoriesWithScenarios.push([runGroup, groupScenarios]);
    }
  });

  return (
    <section>
      <h3 className="text-3xl">{categorizedScenarioNames.size} scenarios</h3>
      <ul>
        {categoriesWithScenarios.map(([category, scenarios]) => (
          <li key={category.name} className="my-3">
            <ReactRouterLink
              className="text-black"
              to={"groups/" + category.name}
            >
              <h2>{category.display_name}</h2>
            </ReactRouterLink>
            <ul className="list-disc list-inside">
              {scenarios.map((scenario) =>
                scenario.todo ? (
                  <li
                    key={scenario.name}
                    className={`${
                      scenario.todo ? "ml-4 text-slate-300" : "ml-4"
                    }`}
                  >
                    {scenario.display_name}
                  </li>
                ) : (
                  <ReactRouterLink
                    className="text-black"
                    to={"groups/" + scenario.name}
                  >
                    <li
                      key={scenario.name}
                      className={`${
                        scenario.todo ? "ml-4 text-slate-300" : "ml-4"
                      }`}
                    >
                      {scenario.display_name}
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
