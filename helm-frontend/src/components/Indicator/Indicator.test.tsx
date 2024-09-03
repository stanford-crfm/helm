import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import Indicator from "./Indicator";

it("renders quasi_exact_match success correctly", () => {
  const { container } = render(
    <Indicator
      stats={{
        quasi_exact_match: 0.5,
      }}
    />,
  );

  expect(screen.getByText("quasi exact match: 0.5"));
  expect(container.getElementsByClassName("bg-green-100").length).toBe(1);
});

it("renders quasi_exact_match fail correctly", () => {
  const { container } = render(
    <Indicator
      stats={{
        quasi_exact_match: 0,
      }}
    />,
  );

  expect(container.getElementsByClassName("bg-red-100").length).toBe(1);
  expect(screen.getByText("quasi exact match: 0"));
});

it("renders toxic_frac success correctly", () => {
  const { container } = render(
    <Indicator
      stats={{
        toxic_frac: 0,
      }}
    />,
  );

  expect(container.getElementsByClassName("bg-green-100").length).toBe(1);
  expect(screen.getByText("toxic frac: 0"));
});

it("renders toxic_frac fail correctly", () => {
  const { container } = render(
    <Indicator
      stats={{
        toxic_frac: 0.5,
      }}
    />,
  );

  expect(container.getElementsByClassName("bg-red-100").length).toBe(1);
  expect(screen.getByText("toxic frac: 0.5"));
});

it("renders exact_match success correctly", () => {
  const { container } = render(
    <Indicator
      stats={{
        exact_match: 1,
      }}
    />,
  );

  expect(container.getElementsByClassName("bg-green-100").length).toBe(1);
  expect(screen.getByText("exact match: 1"));
});

it("renders exact_match fail correctly", () => {
  const { container } = render(
    <Indicator
      stats={{
        exact_match: 0,
      }}
    />,
  );

  expect(container.getElementsByClassName("bg-red-100").length).toBe(1);
  expect(screen.getByText("exact match: 0"));
});
