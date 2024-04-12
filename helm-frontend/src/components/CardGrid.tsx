import { useEffect, useState } from "react";
import ReleaseIndexEntry from "@/types/ProjectMetadata";
import ProjectCard from "./ProjectCard";

export default function CardGrid() {
  const [projectEntries, setProjectEntries] = useState<
    ReleaseIndexEntry[] | undefined
  >();

  useEffect(() => {
    fetch(
      "https://storage.googleapis.com/crfm-helm-public/config/release_index.json",
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
          projectEntries.map((project, index) => (
            <ProjectCard
              key={index}
              id={project.id}
              title={project.title}
              imageUrl={
                project.imageUrl !== undefined
                  ? String(project.imageUrl)
                  : undefined
              }
              text={project.description}
            />
          ))}
      </div>
    </div>
  );
}
