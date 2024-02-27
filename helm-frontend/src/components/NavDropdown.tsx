import { ChevronDownIcon } from "@heroicons/react/24/solid";

function NavDropdown() {
  let currentSubsite = "";
  if (window.HELM_TYPE === "LITE") {
    currentSubsite = "Lite";
  } else if (window.HELM_TYPE === "CLASSIC") {
    currentSubsite = "Classic";
  } else if (window.HELM_TYPE === "HEIM") {
    currentSubsite = "HEIM";
  } else if (window.HELM_TYPE === "INSTRUCT") {
    currentSubsite = "Instruct";
  }

  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="btn normal-case bg-white font-bold p-2 border-0 text-lg"
      >
        {currentSubsite}{" "}
        <ChevronDownIcon fill="black" color="black" className="text w-4 h-4" />
      </div>
      <ul
        tabIndex={0}
        className="dropdown-content z-[1] menu p-1 shadow-lg bg-base-100 rounded-box width-100 block"
      >
        <li>
          <a href="https://crfm.stanford.edu/helm/lite/">
            <strong>Lite:</strong>{" "}
            <span>Lightweight, broad evaluation of recent language models</span>
          </a>
        </li>
        <li>
          <a href="https://crfm.stanford.edu/helm/classic/">
            <strong>Classic:</strong>{" "}
            <span>
              Thorough language model evaluations based on the scenarios from
              the original HELM paper
            </span>
          </a>
        </li>
        <li>
          <a href="https://crfm.stanford.edu/heim/">
            <strong>HEIM:</strong>{" "}
            <span>Holistic evaluation of text-to-image models</span>
          </a>
        </li>
        <li>
          <a href="https://crfm.stanford.edu/helm/instruct/">
            <strong>Instruct:</strong>{" "}
            <span>
              Evaluations of instruction following models with absolute ratings
            </span>
          </a>
        </li>
      </ul>
    </div>
  );
}

export default NavDropdown;
