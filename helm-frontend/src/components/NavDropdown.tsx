import { ChevronDownIcon } from "@heroicons/react/24/solid";

function getCurrentSubsite(): string {
  // TODO: Fetch this from a configuration file.
  if (window.HELM_TYPE === "LITE") {
    return "Lite";
  } else if (window.HELM_TYPE === "CLASSIC") {
    return "Classic";
  } else if (window.HELM_TYPE === "HEIM") {
    return "HEIM";
  } else if (window.HELM_TYPE === "INSTRUCT") {
    return "Instruct";
  }
  return "Lite";
}

function NavDropdown() {
  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="btn normal-case bg-white font-bold p-2 border-0 text-lg"
        aria-haspopup="true"
        aria-controls="menu"
      >
        {getCurrentSubsite()}{" "}
        <ChevronDownIcon fill="black" color="black" className="text w-4 h-4" />
      </div>
      <ul
        tabIndex={0}
        className="-translate-x-36 dropdown-content z-[1] menu p-1 shadow-lg bg-base-100 rounded-box w-max text-base"
        role="menu"
      >
        <li>
          <a
            href="https://crfm.stanford.edu/helm/lite/"
            className="block"
            role="menuitem"
          >
            <strong className="inline">Lite:</strong> Lightweight, broad
            evaluation of the capabilities of language models using in-context
            learning
          </a>
        </li>
        <li>
          <a
            href="https://crfm.stanford.edu/helm/classic/"
            className="block"
            role="menuitem"
          >
            <strong>Classic:</strong> Thorough language model evaluations based
            on the scenarios from the original HELM paper
          </a>
        </li>
        <li>
          <a
            href="https://crfm.stanford.edu/heim/"
            className="block"
            role="menuitem"
          >
            <strong>HEIM:</strong> Holistic evaluation of text-to-image models
          </a>
        </li>
        <li>
          <a
            href="https://crfm.stanford.edu/helm/instruct/"
            className="block"
            role="menuitem"
          >
            <strong>Instruct:</strong> Evaluations of instruction following
            models with absolute ratings
          </a>
        </li>
      </ul>
    </div>
  );
}

export default NavDropdown;
