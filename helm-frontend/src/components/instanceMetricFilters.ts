import type Instance from "@/types/Instance";
import type DisplayPrediction from "@/types/DisplayPrediction";

export type DisplayPredictionsMap = Record<
  string,
  Record<string, DisplayPrediction[]>
>;

export type MetricOutcomeFilter = "all" | "correct" | "incorrect";

const BINARY_METRIC_VALUES = new Set([0, 1]);

function getPredictionsForInstance(
  displayPredictionsMap: DisplayPredictionsMap,
  instanceId: string,
): DisplayPrediction[] {
  return Object.values(displayPredictionsMap[instanceId] ?? {}).flat();
}

export function getFilterableMetricKeys(
  displayPredictionsMap: DisplayPredictionsMap,
): string[] {
  const metricValues = new Map<string, Set<number>>();

  Object.values(displayPredictionsMap).forEach((predictionsByPerturbation) => {
    Object.values(predictionsByPerturbation).forEach((predictions) => {
      predictions.forEach((prediction) => {
        Object.entries(prediction.stats).forEach(([metricKey, metricValue]) => {
          if (typeof metricValue !== "number") {
            return;
          }
          const values = metricValues.get(metricKey) ?? new Set<number>();
          values.add(metricValue);
          metricValues.set(metricKey, values);
        });
      });
    });
  });

  return Array.from(metricValues.entries())
    .filter(([, values]) => {
      return (
        values.size > 0 &&
        Array.from(values).every((value) => BINARY_METRIC_VALUES.has(value))
      );
    })
    .map(([metricKey]) => metricKey);
}

export function filterInstancesByMetric(
  instances: Instance[],
  displayPredictionsMap: DisplayPredictionsMap,
  selectedMetric: string,
  selectedOutcome: MetricOutcomeFilter,
): Instance[] {
  if (!selectedMetric || selectedOutcome === "all") {
    return instances;
  }

  const expectedValue = selectedOutcome === "correct" ? 1 : 0;

  return instances.filter((instance) => {
    return getPredictionsForInstance(displayPredictionsMap, instance.id).some(
      (prediction) => prediction.stats[selectedMetric] === expectedValue,
    );
  });
}
