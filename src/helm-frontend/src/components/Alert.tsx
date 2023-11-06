import { useState } from "react";
import Link from "./Link";

export default function Alert() {
  const [visible, setVisible] = useState(true);

  const handleClose = () => {
    setVisible(false);
  };

  return (
    visible && (
      <div
        className="fixed bottom-5 right-5 bg-gray-100 border border-gray-400 text-gray-700 px-4 py-3 rounded z-50"
        role="alert"
      >
        <div className="px-3">
          <strong className="font-bold">
            Welcome to the new results view,
          </strong>
          <span className="block sm:inline"> for the old view, </span>
          <Link to={"/groups"}>
            <a className="underline text-gray-700 mr-2">click here</a>
          </Link>
        </div>
        <span
          className="absolute top-1 bottom-1 right-0 px-4 py-3"
          onClick={handleClose}
        >
          <img
            src="https://www.svgrepo.com/show/12848/x-symbol.svg"
            alt="Close"
            className="h-3 w-3"
          />
        </span>
      </div>
    )
  );
}
