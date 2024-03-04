import { ChevronDownIcon } from "@heroicons/react/24/solid";
import ReleaseIndexEntry from "@/types/ReleaseIndexEntry";
import { useEffect, useState } from "react";
import getReleaseUrl from "@/utils/getReleaseUrl";

function NavDropdown() {
  const [releaseIndex, setReleaseIndex] = useState<ReleaseIndexEntry[]>([]);
  const [currReleaseIndexEntry, setCurrReleaseIndexEntry] = useState<
    ReleaseIndexEntry | undefined
  >();

  useEffect(() => {
    fetch("/releaseIndex.json")
      .then((response) => response.json())
      .then((data: ReleaseIndexEntry[]) => {
        setReleaseIndex(data);
        // set currReleaseIndexEntry to val where releaseIndexEntry.id matches window.RELEASE_INDEX_ID
        if (window.RELEASE_INDEX_ID) {
          const currentEntry = data.find(
            (entry) => entry.id === window.RELEASE_INDEX_ID,
          );
          setCurrReleaseIndexEntry(currentEntry);
          // handles falling back to HELM lite as was previously done in this file
        } else {
          const currentEntry = data.find((entry) => entry.id === "lite");
          setCurrReleaseIndexEntry(currentEntry);
        }
      })
      .catch((error) => {
        console.error("Error fetching JSON:", error);
      });
  }, []);

  function getCurrentSubsite(): string {
    return currReleaseIndexEntry !== undefined &&
      currReleaseIndexEntry.title !== undefined
      ? currReleaseIndexEntry.title
      : "Lite";
  }

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
        {releaseIndex.map((item, index) => (
          <li key={index}>
            <a
              href={getReleaseUrl(item.id, item)}
              className="block"
              role="menuitem"
            >
              <strong>{item.title}:</strong> {item.description}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default NavDropdown;
