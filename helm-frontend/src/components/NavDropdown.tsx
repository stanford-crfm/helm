import { ChevronDownIcon } from "@heroicons/react/24/solid";
import ProjectMetadata from "@/types/ProjectMetadata";
import { useEffect, useState } from "react";
import getReleaseUrl from "@/utils/getReleaseUrl";

function NavDropdown() {
  const [projectMetadata, setProjectMetadata] = useState<ProjectMetadata[]>([]);
  const [currProjectMetadata, setCurrProjectMetadata] = useState<
    ProjectMetadata | undefined
  >();

  useEffect(() => {
    fetch(
      "https://storage.googleapis.com/crfm-helm-public/config/project_metadata.json",
    )
      .then((response) => response.json())
      .then((data: ProjectMetadata[]) => {
        setProjectMetadata(data);
        // set currProjectMetadata to val where projectMetadataEntry.id matches window.PROJECT_ID
        if (window.PROJECT_ID) {
          if (window.PROJECT_ID === "global") {
            // TODO replace this hardcoding if we choose to put global in the project metadata array
            setCurrProjectMetadata({
              id: "global",
              title: "All Projects",
              description: "description",
              releases: ["releases"],
              imageUrl: "imageUrl",
            });
          } else {
            const currentEntry = data.find(
              (entry) => entry.id === window.PROJECT_ID,
            );
            setCurrProjectMetadata(currentEntry);
          }
          // handles falling back to HELM lite as was previously done in this file
        } else {
          const currentEntry = data.find((entry) => entry.id === "lite");
          setCurrProjectMetadata(currentEntry);
        }
      })
      .catch((error) => {
        console.error("Error fetching JSON:", error);
      });
  }, []);

  if (
    currProjectMetadata === undefined ||
    currProjectMetadata.title === undefined
  ) {
    return null;
  }

  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="btn normal-case bg-white font-bold p-2 border-0 text-lg block whitespace-nowrap"
        aria-haspopup="true"
        aria-controls="menu"
      >
        {currProjectMetadata.title}&nbsp;
        <ChevronDownIcon
          fill="black"
          color="black"
          className="text w-4 h-4 inline"
        />
      </div>
      <ul
        tabIndex={0}
        className="-translate-x-36 dropdown-content z-[1] menu p-1 shadow-lg bg-base-100 rounded-box w-max text-base"
        role="menu"
      >
        {projectMetadata.map((item, index) => (
          <li key={index}>
            <a
              href={getReleaseUrl(undefined, item.id)}
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
