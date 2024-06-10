import getReleaseUrl from "@/utils/getReleaseUrl";
import React from "react";

interface CardProps {
  id: string;
  title: string;
  text: string;
}

const ProjectCard: React.FC<CardProps> = ({ id, title, text }) => {
  if (!title.includes("HE")) {
    title = "HELM " + title;
  }
  return (
    <div className="max-w-sm rounded overflow-hidden bg-gray-100 hover:scale-105 transition-transform duration-300">
      <a href={getReleaseUrl(undefined, id)}>
        <div className="px-6 py-4">
          <div className="font-bold text-xl mb-2">
            <div className="py-3">
              <svg
                fill="#000000"
                width="20px"
                height="20px"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M22,7H16.333V4a1,1,0,0,0-1-1H8.667a1,1,0,0,0-1,1v7H2a1,1,0,0,0-1,1v8a1,1,0,0,0,1,1H22a1,1,0,0,0,1-1V8A1,1,0,0,0,22,7ZM7.667,19H3V13H7.667Zm6.666,0H9.667V5h4.666ZM21,19H16.333V9H21Z" />
              </svg>
            </div>
            {title + " →"}
          </div>
          <p className="text-gray-700 text-base">{text}</p>
        </div>
      </a>
    </div>
  );
};

export default ProjectCard;
