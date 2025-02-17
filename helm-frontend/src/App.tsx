import "./App.css";
import { HashRouter as Router, Route, Routes } from "react-router-dom";
import Layout from "@/layouts/Main";
import Models from "@/routes/Models";
import Scenarios from "@/routes/Scenarios";
import Groups from "@/routes/Groups";
import Runs from "@/routes/Runs";
import Run from "@/routes/Run";
import Leaderboard from "@/routes/Leaderboard";
import Home from "@/routes/Home";

export default function App() {
  /* NOTE: groups/:groupName is a legacy route. Links to it were removed on 2025-02-07.
           The route itself is temporarily retained for backwards compatibility.
  */
  return (
    <Router>
      <Routes>
        <Route path={`/`} element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="leaderboard" element={<Leaderboard />} />
          <Route path="leaderboard/:groupName" element={<Leaderboard />} />
          <Route path="models" element={<Models />} />
          <Route path="scenarios" element={<Scenarios />} />
          <Route path="groups" element={<Groups />} />
          <Route path="groups/:groupName" element={<Leaderboard />} />
          <Route path="runs" element={<Runs />} />
          <Route path="runs/:runName" element={<Run />} />
        </Route>
      </Routes>
    </Router>
  );
}
