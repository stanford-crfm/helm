import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import AccessBadge from "./AccessBadge";

test("displays access badge", () => {
  render(
    <>
      <AccessBadge level="open" />
      <AccessBadge level="limited" />
      <AccessBadge level="closed" />
    </>,
  );

  expect(screen.getByText("Open").parentElement).toHaveClass("bg-green-100");
  expect(screen.getByText("Limited").parentElement).toHaveClass(
    "bg-yellow-100",
  );
  expect(screen.getByText("Closed").parentElement).toHaveClass("bg-red-100");
});
