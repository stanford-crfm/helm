import { useRef } from "react";

interface Props {
  runName: string;
  onAgree: () => void;
}

export default function Tab({ runName, onAgree }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleAgreement = () => {
    if (
      inputRef.current !== null &&
      inputRef.current.value.trim() === "Yes, I agree"
    ) {
      onAgree();
    } else {
      alert("Please type 'Yes, I agree' exactly.");
    }
  };
  const agreement = runName.includes("gpqa") ? (
    <GPQATerms />
  ) : runName.includes("ewok") ? (
    <EWoKTerms />
  ) : null;

  return (
    <div className="mb-8">
      {agreement}
      <p className="mb-4">
        If you agree to this condition, please type{" "}
        <strong>"Yes, I agree"</strong> in the box below and then click{" "}
        <strong>Decrypt</strong>.
      </p>
      <div className="flex gap-2 mt-2">
        <input
          type="text"
          ref={inputRef}
          className="input input-bordered"
          placeholder='Type "Yes, I agree"'
        />
        <button onClick={handleAgreement} className="btn btn-primary">
          Decrypt
        </button>
      </div>
      <hr className="my-4" />
    </div>
  );
}

function GPQATerms() {
  return (
    <div>
      <p className="mb-4">
        The GPQA dataset instances are encrypted by default to comply with the
        following request:
      </p>
      <blockquote className="italic border-l-4 border-gray-300 pl-4 text-gray-700 mb-4">
        “We ask that you do not reveal examples from this dataset in plain text
        or images online, to minimize the risk of these instances being included
        in foundation model training corpora.”
      </blockquote>
    </div>
  );
}

function EWoKTerms() {
  return (
    <div>
      <p className="mb-4">
        The EWoK dataset instances are encrypted by default to comply with the
        following request:
      </p>
      <blockquote className="italic border-l-4 border-gray-300 pl-4 text-gray-700 mb-4">
        “PLEASE DO NOT distribute any of the EWoK materials or derivatives
        publicly in plain-text! Any materials should appear in
        password-protected ZIP files or behind gated authentication mechanisms
        such as Huggingface datasets.”
      </blockquote>
    </div>
  );
}
