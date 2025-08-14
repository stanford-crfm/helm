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
        <div
          className={
            "bg-base-200 p-2 block overflow-auto w-full max-h-[36rem] mb-2 whitespace-pre-wrap localize-text-direction"
          }
        >
          {value}
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
        className="modal p-16 bg-opacity-80 bg-white"
        onClick={() => setShowModal(false)}
      >
        <div
          className={
            "modal-box max-w-none p-4 whitespace-pre-wrap bg-base-200 localize-text-direction"
          }
        >
          {value}
        </div>
      </dialog>
    </>
  );
}
