import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import MarkdownValue from "./MarkdownValue";

it("renders correctly", () => {
  render(<MarkdownValue value="**bold**" />);

  expect(screen.getByText("bold"));
});

it("should add a class", () => {
  render(<MarkdownValue value="[test](https://example.com)" />);

  expect(screen.getByText("test")).toHaveClass("link");
});
