import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import type DisplayPrediction from "@/types/DisplayPrediction";
import type DisplayRequest from "@/types/DisplayRequest";
import type MetricFieldMap from "@/types/MetricFieldMap";
import Predictions from "./Predictions";

it("renders instance metrics in a compact two-column list", () => {
  const predictions: DisplayPrediction[] = [
    {
      instance_id: "instance-1",
      predicted_text: "hello world",
      train_trial_index: 0,
      stats: {
        very_long_metric_name_for_layout_regression: 0.75,
        exact_match: 1,
      },
      mapped_output: undefined,
    },
  ];
  const requests: DisplayRequest[] = [
    {
      instance_id: "instance-1",
      train_trial_index: 0,
      request: {
        echo_prompt: false,
        embedding: false,
        frequency_penalty: 0,
        max_tokens: 16,
        model: "example/model",
        num_completions: 1,
        presence_penalty: 0,
        prompt: "Prompt text",
        stop_sequences: [],
        temperature: 0,
        top_k_per_token: 1,
        top_p: 1,
        multimodal_prompt: undefined,
        messages: undefined,
      },
    },
  ];
  const metricFieldMap: MetricFieldMap = {
    very_long_metric_name_for_layout_regression: {
      name: "very_long_metric_name_for_layout_regression",
      display_name: "Very long metric name for layout regression",
      short_display_name: "Very long metric",
      description: "Long label used to verify compact metric layout rendering.",
      lower_is_better: false,
    },
    exact_match: {
      name: "exact_match",
      display_name: "Exact match",
      short_display_name: "EM",
      description: "Exact match metric",
      lower_is_better: false,
    },
  };

  render(
    <Predictions
      predictions={predictions}
      requests={requests}
      metricFieldMap={metricFieldMap}
    />,
  );

  const metricsContainer = screen.getByTestId("prediction-metrics");
  expect(metricsContainer).toHaveClass("max-w-3xl");
  expect(
    screen.getByText("Very long metric name for layout regression"),
  ).toBeInTheDocument();
  expect(screen.getByText("0.75")).toBeInTheDocument();
  expect(screen.getByText("Exact match")).toBeInTheDocument();
});
