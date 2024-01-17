import { useState } from "react";
import { ArrowsPointingOutIcon } from "@heroicons/react/24/solid";

interface Props {
  value: string;
}

export default function Preview({ value }: Props) {
  const [showButton, setShowButton] = useState<boolean>(false);
  const [showModal, setShowModal] = useState<boolean>(false);

  return (
    <>
      <div
        onMouseOver={() => setShowButton(true)}
        onMouseOut={() => setShowButton(false)}
        className="relative"
      >
        <div className="bg-base-200 p-2 block overflow-auto w-full max-h-72 mb-2">
          <pre>{value}</pre>
        </div>

        {showButton ? (
          <button
            className="bg-white absolute p-2 leading-none height-fit min-h-none right-1 bottom-1 shadow"
            onClick={() => setShowModal(true)}
          >
            <ArrowsPointingOutIcon
              fill="black"
              color="black"
              className="text w-4 h-4"
            />
          </button>
        ) : null}
      </div>
      <dialog
        open={showModal}
        className="modal p-16"
        onClick={() => setShowModal(false)}
      >
        <div className="modal-box max-w-none p-16">
          <div>
            <pre className="p-6">{value}</pre>
          </div>
        </div>
      </dialog>
    </>
  );
}
