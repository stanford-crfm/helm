import React from "react";

interface CardProps {
  title: string;
  imageUrl: string | undefined;
  text: string;
}

const ProjectCard: React.FC<CardProps> = ({ title, imageUrl, text }) => {
  return (
    <div className="max-w-sm rounded overflow-hidden bg-gray-100">
      {imageUrl ? <img className="w-full" src={imageUrl} alt={title} /> : <></>}
      <div className="px-6 py-4">
        <div className="font-bold text-xl mb-2">{title}</div>
        <p className="text-gray-700 text-base">{text}</p>
      </div>
    </div>
  );
};

export default ProjectCard;
