import { Outlet } from "react-router-dom";
import NavBar from "@/components/NavBar/NavBar";
import MinimalNavBar from "@/components/MinimalNavBar";

export default function Main() {
  return (
    <>
      {window.PROJECT_ID === "home" ? <MinimalNavBar /> : <NavBar />}
      <main className="p-8 pt-0">
        <div className="mx-auto max-w-[1500]px">
          <Outlet />
        </div>
      </main>
    </>
  );
}
