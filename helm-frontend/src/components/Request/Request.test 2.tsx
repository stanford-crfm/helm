import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import DisplayRequest from "@/types/DisplayRequest";
import Request from "./Request";

it("renders correctly", () => {
  const request: DisplayRequest = {
    instance_id: "id1662",
    train_trial_index: 0,
    request: {
      echo_prompt: false,
      embedding: false,
      frequency_penalty: 0,
      max_tokens: 5,
      model: "AlephAlpha/luminous-base",
      num_completions: 1,
      presence_penalty: 0,
      prompt: "Passage: Mice are afraid of cats.",
      stop_sequences: ["\n"],
      temperature: 0,
      top_k_per_token: 1,
      top_p: 1,
      multimodal_prompt: undefined,
      messages: undefined,
    },
  };

  render(<Request request={request} />);

  /* TODO: reimplement this: expect(screen.getByText("Prompt"));*/
  expect(screen.getAllByText("Passage: Mice are afraid of cats.").length).toBe(
    2,
  );
});
