import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { MemoryRouter } from "react-router-dom";
import NavBar from "./NavBar";

test("displays nav bar", () => {
  render(
    <MemoryRouter>
      <NavBar />
    </MemoryRouter>,
  );

  expect(screen.getByRole("navigation")).toHaveTextContent(
    "LeaderboardModelsScenariosPredictionsGitHubLeaderboardModelsScenariosPredictionsGitHub",
  );
});
