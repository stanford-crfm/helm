import "./App.css";
import { HashRouter as Router, Route, Routes } from "react-router-dom";
import Layout from "@/layouts/Main";
import Models from "@/routes/Models";
import Scenarios from "@/routes/Scenarios";
import Groups from "@/routes/Groups";
import Group from "@/routes/Group";
import Runs from "@/routes/Runs";
import Run from "@/routes/Run";
import Leaderboard from "@/routes/Leaderboard";
import Home from "@/routes/Home";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path={`/`} element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="leaderboard" element={<Leaderboard />} />
          <Route path="models" element={<Models />} />
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
