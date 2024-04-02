import getReleaseUrl from "@/utils/getReleaseUrl";
import React from "react";

interface CardProps {
  id: string;
  title: string;
  imageUrl: string | undefined;
  text: string;
}

const ProjectCard: React.FC<CardProps> = ({ id, title, imageUrl, text }) => {
  if (!title.includes("HE")) {
    title = "HELM " + title;
  }
  return (
    <div className="max-w-sm rounded overflow-hidden bg-gray-100 hover:scale-105 transition-transform duration-300">
      {imageUrl ? <img className="w-full" src={imageUrl} alt={title} /> : <></>}
      <div className="px-6 py-4">
        <div className="font-bold text-xl mb-2">
          <a href={getReleaseUrl(undefined, id)}> {title + " →"}</a>
        </div>
        <p className="text-gray-700 text-base">{text}</p>
      </div>
    </div>
  );
};

export default ProjectCard;
