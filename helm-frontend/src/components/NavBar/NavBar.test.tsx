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
    "LeaderboardModelsScenariosPredictionsGitHubLite Lite: Lightweight, broad evaluation of the capabilities of language models using in-context learningClassic: Thorough language model evaluations based on the scenarios from the original HELM paperHEIM: Holistic evaluation of text-to-image modelsInstruct: Evaluations of instruction following models with absolute ratingsLeaderboardModelsScenariosPredictionsGitHubRelease unknown () v1.1.0v1.0.0",
  );
});
