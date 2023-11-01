import "./App.css";
import {
  HashRouter as Router,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import Layout from "@/layouts/Main";
import Models from "@/routes/Models";
import Scenarios from "@/routes/Scenarios";
import Groups from "@/routes/Groups";
import Group from "@/routes/Group";
import Runs from "@/routes/Runs";
import Run from "@/routes/Run";
import Landing from "@/routes/Landing";
import Leaderboard from "@/routes/Leaderboard";
import { useEffect } from "react";

function RedirectToBasePath() {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
  const legacyRelease = (window as any).LEGACY_RELEASE;
  const location = useLocation();

  useEffect(() => {
    if (location.pathname === "/") {
      window.location.replace(`#/${legacyRelease}`);
    }
  }, [location, legacyRelease]);

  return null;
}

export default function App() {
  return (
    <Router>
      <RedirectToBasePath />
      <Routes>
        <Route path={`/:legacyRelease`} element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="models" element={<Models />} />
          <Route path="leaderboard" element={<Leaderboard />} />
          <Route path="scenarios" element={<Scenarios />} />
          <Route path="groups" element={<Groups />} />
          <Route path="groups/:groupName" element={<Group />} />
          <Route path="runs" element={<Runs />} />
          <Route path="runs/:runName" element={<Run />} />
        </Route>
      </Routes>
    </Router>
  );
}
