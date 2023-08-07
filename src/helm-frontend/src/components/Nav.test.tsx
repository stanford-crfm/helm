import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { MemoryRouter } from "react-router-dom";
import Nav from "./Nav";

test("displays nav", () => {
  render(
    <MemoryRouter>
      <Nav />
    </MemoryRouter>,
  );

  expect(screen.getByRole("navigation")).toHaveTextContent(
    "ModelsScenariosResultsRaw RunsModelsScenariosResultsRaw Runs",
  );
});
