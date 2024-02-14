import { useState } from "react";
import { Link } from "react-router-dom";

function NavDropdown() {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="relative inline-block text-left p-2">
      <div className="inline-flex items-center p-2">
        <Link to="/">
          <div className="flex items-center">
            <img
              src="https://crfm.stanford.edu/helm/v0.3.0/images/helm-logo-simple.png"
              alt="Image 1"
              className="h-10 object-cover"
            />
            {/* Manually set whether Classic or Not via config, otherwise don't show this */}
            {window.HELM_TYPE ? (
              <div className="hidden lg:flex pl-3">
                {" "}
                <strong>{window.HELM_TYPE}</strong>{" "}
              </div>
            ) : (
              <> </>
            )}
          </div>
        </Link>

        {/* Chevron Button */}
        <button
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className="inline-flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 ml-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>

      {dropdownOpen && (
        <div className="absolute mt-2 w-max translate-x-4 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5">
          <div
            className="py-1"
            role="menu"
            aria-orientation="vertical"
            aria-labelledby="options-menu"
          >
            <div
              className="block px-4 py-2 text-md text-gray-700 hover:bg-gray-100 hover:text-gray-900"
              role="menuitem"
            >
              <a href="https://crfm.stanford.edu/helm/classic/latest/">
                <div className="flex items-center">
                  <span>
                    <strong>HELM Classic: </strong>Thorough language model
                    evaluations based on the scenarios from the original HELM
                    paper
                  </span>
                </div>
              </a>
            </div>

            <div
              className="block px-4 py-2 text-md text-gray-700 hover:bg-gray-100 hover:text-gray-900"
              role="menuitem"
            >
              <a href="https://crfm.stanford.edu/helm/lite/latest/">
                <div className="flex items-center">
                  <span>
                    <strong>HELM Lite: </strong>Lightweight, broad evaluation of
                    the capabilities of language models using in-context
                    learning
                  </span>
                </div>
              </a>
            </div>
            <div
              className="block px-4 py-2 text-md text-gray-700 hover:bg-gray-100 hover:text-gray-900"
              role="menuitem"
            >
              <a href="https://crfm.stanford.edu/heim/latest/">
                <div className="flex items-center">
                  <span>
                    <strong>HEIM: </strong>Holistic evaluation of text-to-image
                    models
                  </span>
                </div>
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default NavDropdown;
