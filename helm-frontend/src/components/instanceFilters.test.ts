import "@testing-library/jest-dom";
import { describe, expect, it } from "vitest";

import type Instance from "@/types/Instance";

import { matchesInstanceSearch } from "./instanceFilters";

const baseInstance: Instance = {
  id: "instance-42",
  split: "test",
  input: {
    text: "The capital of France is Paris.",
    multimedia_content: undefined,
    messages: undefined,
  },
  references: [{ output: { text: "Paris" }, tags: [] }],
};

describe("matchesInstanceSearch", () => {
  it("matches the instance id", () => {
    expect(matchesInstanceSearch(baseInstance, "instance-42")).toBe(true);
  });

  it("matches the input text case-insensitively", () => {
    expect(matchesInstanceSearch(baseInstance, "capital of france")).toBe(true);
  });

  it("matches reference text", () => {
    expect(matchesInstanceSearch(baseInstance, "paris")).toBe(true);
  });

  it("returns true for an empty query", () => {
    expect(matchesInstanceSearch(baseInstance, "")).toBe(true);
  });

  it("returns false for unrelated text", () => {
    expect(matchesInstanceSearch(baseInstance, "tokyo")).toBe(false);
  });
});
