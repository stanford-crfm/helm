import "./App.css";
import { HashRouter as Router, Route, Routes } from "react-router-dom";
import Layout from "@/layouts/Main";
import Home from "@/routes/Home";
import Models from "@/routes/Models";
import Scenarios from "@/routes/Scenarios";
import Groups from "@/routes/Groups";
import Group from "@/routes/Group";
import Runs from "@/routes/Runs";
import Run from "@/routes/Run";
import LegacyLanding from "@/routes/LegacyLanding";
export default function App() {
	return (
		<Router>
			<Routes>
				<Route element={<Layout />}>
					<Route path={"/"} element={<Home />} />
					<Route path="/models" element={<Models />} />
					<Route path="/landing" element={<LegacyLanding />} />
					<Route path="/scenarios" element={<Scenarios />} />
					<Route path="/groups" element={<Groups />} />
					<Route path="/groups/:groupName" element={<Group />} />
					<Route path="/runs" element={<Runs />} />
					<Route path="/runs/:runName" element={<Run />} />
				</Route>
			</Routes>
		</Router>
	);
}
