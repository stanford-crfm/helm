import helmHero from "@/assets/helmhero.png";
import { Link } from "react-router-dom";

export default function SimpleHero() {
  return (
    <div className="flex px-6 py-20">
      <div className="flex-1 p-4 flex flex-col justify-center">
        <div className="flex justify-start">
          <h1 className="text-5xl mb-4 mx-4 mt-2">
            <strong>
              A holistic framework for evaluating foundation models.
            </strong>
          </h1>
        </div>
        <div className="flex justify-start mt-6 ml-4">
          <button
            className="px-6 btn btn-grey rounded-md"
            onClick={() =>
              window.scrollTo({
                top: 700,
                behavior: "smooth",
              })
            }
          >
            <body>Projects</body>
          </button>
          <Link to="https://github.com/stanford-crfm/helm" className="ml-4">
            <button className="px-6 btn btn-grey rounded-md">Github</button>
          </Link>
        </div>
      </div>

      <div className="w-1/3 mx-4">
        <img
          src={helmHero}
          alt="HELM Hero"
          className="object-cover w-full h-full"
        />
      </div>
    </div>
  );
}
