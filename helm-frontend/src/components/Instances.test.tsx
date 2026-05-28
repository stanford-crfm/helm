import { MemoryRouter, Route, Routes } from "react-router-dom";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { describe, expect, it, vi } from "vitest";
import Instances from "./Instances";

vi.mock("@/components/Loading", () => ({
  default: () => <div>Loading...</div>,
}));

vi.mock("@/components/InstanceData", () => ({
  default: () => <div>Instance Data</div>,
}));

vi.mock("@/components/Pagination", () => ({
  default: ({
    currentPage,
    totalPages,
  }: {
    currentPage: number;
    totalPages: number;
  }) => (
    <div>
      Pagination {currentPage}/{totalPages}
    </div>
  ),
}));

const instances = Array.from({ length: 15 }, (_, idx) => ({
  id: `id${idx + 1}`,
  input: { text: `input-${idx + 1}` },
  references: [],
  split: "test",
}));

vi.mock("@/services/getInstances", () => ({
  default: vi.fn(async () => instances),
}));

vi.mock("@/services/getDisplayPredictionsByName", () => ({
  default: vi.fn(
    async () =>
      instances.map((instance) => ({
        instance_id: instance.id,
        predicted_text: "",
        train_trial_index: 0,
        stats: {},
        mapped_output: undefined,
        perturbation: undefined,
      })),
  ),
}));

vi.mock("@/services/getDisplayRequestsByName", () => ({
  default: vi.fn(
    async () =>
      instances.map((instance) => ({
        instance_id: instance.id,
        train_trial_index: 0,
        request: {
          echo_prompt: false,
          embedding: false,
          frequency_penalty: 0,
          max_tokens: 1,
          model: "test-model",
          num_completions: 1,
          presence_penalty: 0,
          prompt: "",
          stop_sequences: [],
          temperature: 0,
          top_k_per_token: 1,
          top_p: 1,
          multimodal_prompt: undefined,
          messages: undefined,
        },
        perturbation: undefined,
      })),
  ),
}));

describe("Instances", () => {
  it("respects the instancesPage query parameter on initial render", async () => {
    render(
      <MemoryRouter initialEntries={["/runs/test?instancesPage=2&instance=id12"]}>
        <Routes>
          <Route
            path="/runs/:id"
            element={
              <Instances
                runName="test-run"
                suite="test-suite"
                metricFieldMap={{}}
                userAgreed
              />
            }
          />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Instance id: id11 [split: test]")).toBeInTheDocument();
    expect(screen.getByText("Instance id: id12 [split: test]")).toBeInTheDocument();
    expect(screen.queryByText("Instance id: id1 [split: test]")).not.toBeInTheDocument();
    expect(screen.getByText("Pagination 2/2")).toBeInTheDocument();
  });
});
