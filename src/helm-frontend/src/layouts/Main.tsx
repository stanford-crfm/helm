import { Outlet } from "react-router-dom";
import Nav from "@/components/NavBar/NavBar";

export default function Main() {
  return (
    <>
      <Nav />
      <main className="p-8 md:pt-12 pt-0">
        <div className="mx-auto max-w-[1500]px">
          <Outlet />
        </div>
      </main>
    </>
  );
}
