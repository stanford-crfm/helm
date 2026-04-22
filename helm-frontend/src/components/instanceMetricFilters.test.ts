import { describe, expect, it } from "vitest";
import type Instance from "@/types/Instance";
import type DisplayPrediction from "@/types/DisplayPrediction";
import {
  filterInstancesByMetric,
  getFilterableMetricKeys,
  type DisplayPredictionsMap,
} from "./instanceMetricFilters";

const instances: Instance[] = [
  {
    id: "instance-1",
    input: { text: "Question one" },
    references: [],
    split: "test",
  },
  {
    id: "instance-2",
    input: { text: "Question two" },
    references: [],
    split: "test",
  },
];

function createPrediction(
  instanceId: string,
  stats: DisplayPrediction["stats"],
): DisplayPrediction {
  return {
    instance_id: instanceId,
    predicted_text: "prediction",
    train_trial_index: 0,
    stats,
    mapped_output: undefined,
  };
}

describe("instanceMetricFilters", () => {
  it("discovers binary metrics that can be used as filters", () => {
    const displayPredictionsMap: DisplayPredictionsMap = {
      "instance-1": {
        "": [
          createPrediction("instance-1", {
            exact_match: 1,
            score: 0.75,
          }),
        ],
      },
      "instance-2": {
        "": [
          createPrediction("instance-2", {
            exact_match: 0,
            score: 0.2,
          }),
        ],
      },
    };

    expect(getFilterableMetricKeys(displayPredictionsMap)).toEqual([
      "exact_match",
    ]);
  });

  it("filters down to correct or incorrect instances for a selected metric", () => {
    const displayPredictionsMap: DisplayPredictionsMap = {
      "instance-1": {
        "": [createPrediction("instance-1", { exact_match: 1 })],
      },
      "instance-2": {
        "": [createPrediction("instance-2", { exact_match: 0 })],
      },
    };

    expect(
      filterInstancesByMetric(
        instances,
        displayPredictionsMap,
        "exact_match",
        "correct",
      ).map((instance) => instance.id),
    ).toEqual(["instance-1"]);

    expect(
      filterInstancesByMetric(
        instances,
        displayPredictionsMap,
        "exact_match",
        "incorrect",
      ).map((instance) => instance.id),
    ).toEqual(["instance-2"]);
  });
});
