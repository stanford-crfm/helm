import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import PageTitle from "./PageTitle";

test("display page title", () => {
  render(<PageTitle title="Testing" />);

  expect(screen.getByRole("heading")).toHaveTextContent("Testing");
});

test("display page subtitle", () => {
  render(<PageTitle title="Main Title" subtitle="Sub Title" />);

  const heading = screen.getAllByRole("heading");
  expect(heading).toHaveLength(2);
  expect(heading[0]).toHaveTextContent("Main Title");
  expect(heading[1]).toHaveTextContent("Sub Title");
});
