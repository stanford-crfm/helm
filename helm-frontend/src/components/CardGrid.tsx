import { useEffect, useState } from "react";
import ReleaseIndexEntry from "@/types/ProjectMetadata";
import ProjectCard from "./ProjectCard";

export default function CardGrid() {
  const [projectEntries, setProjectEntries] = useState<
    ReleaseIndexEntry[] | undefined
  >();

  useEffect(() => {
    fetch(
      "https://raw.githubusercontent.com/stanford-crfm/helm/main/helm-frontend/project_metadata.json",
    )
      .then((response) => response.json())
      .then((data: ReleaseIndexEntry[]) => {
        setProjectEntries(data);
      })
      .catch((error) => {
        console.error("Error fetching JSON:", error);
      });
  }, []);

  return (
    <div className="p-10 mb-20">
      <div className="grid grid-cols-3 gap-4">
        {projectEntries &&
          projectEntries.map((project, index) =>
            project.id === "home" ? null : (
              <ProjectCard
                key={index}
                id={project.id}
                title={project.title}
                text={project.description}
              />
            ),
          )}
      </div>
    </div>
  );
}
