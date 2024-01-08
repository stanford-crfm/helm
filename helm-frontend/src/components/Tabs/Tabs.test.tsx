import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import Tabs from "./Tabs";
import Tab from "../Tab";

test("display tabs", () => {
  render(
    <Tabs>
      <Tab onClick={() => {}}>Link One</Tab>
      <Tab onClick={() => {}}>Link Two</Tab>
      <Tab onClick={() => {}} active={true}>
        Link Three
      </Tab>
    </Tabs>,
  );

  expect(screen.getByRole("navigation").childElementCount).toBe(3);
  expect(screen.getByText("Link One")).not.toHaveClass("tab-active");
  expect(screen.getByText("Link Two")).not.toHaveClass("tab-active");
  //expect(screen.getByText("Link Three")).toHaveClass("tab-active");
});
