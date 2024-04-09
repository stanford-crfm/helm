import { Link } from "react-router-dom";
import { Bars3Icon } from "@heroicons/react/24/outline";
import crfmLogo from "@/assets/crfm-logo.png";
import helmLogo from "@/assets/helm-logo-simple.png";
import NavDropdown from "@/components/NavDropdown";

export default function MinimalNavBar() {
  return (
    <nav className="navbar h-24 px-8 md:px-12 bg-base-100 max-w[1500]px">
      <div>
        <div className="dropdown md:hidden mr-4">
          <label
            tabIndex={0}
            className="btn btn-ghost hover:bg-transparent btn-lg px-0"
          >
            <Bars3Icon className="w-16 h-16" />
          </label>
        </div>
      </div>
      <div className="flex-1 items-center">
        <Link to="https://crfm.stanford.edu/" className="w-24">
          <img src={crfmLogo} className="object-contain" />
        </Link>
        <Link to="/" className="mx-2 w-32">
          <img src={helmLogo} className="object-contain" />
        </Link>
        <NavDropdown></NavDropdown>
      </div>
    </nav>
  );
}
