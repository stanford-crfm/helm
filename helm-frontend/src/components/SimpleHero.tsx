import helmHero from "@/assets/helmhero.png";

export default function SimpleHero() {
  return (
    <div className="flex flex-col md:flex-row px-6 py-32">
      <div className="flex-1 p-4 flex flex-col justify-center">
        <div className="flex justify-start">
          <div>
            <h1 className="text-4xl mb-4 mx-4 mt-2">
              <strong>
                A reproducible and transparent framework for evaluating
                foundation models.
              </strong>
            </h1>
            <h3
              className="text-xl
             mb-4 mx-4 mt-2"
            >
              Find leaderboards with many scenarios, metrics, and models with
              support for multimodality and model-graded evaluation.
            </h3>
          </div>
        </div>
        <div className="flex flex-col md:flex-row justify-start mt-6 ml-4">
          <button
            className="px-6 btn btn-grey rounded-md mb-4 md:mb-0"
            onClick={() =>
              window.scrollTo({
                top: 760,
                behavior: "smooth",
              })
            }
          >
            <div>Leaderboards ↓</div>
          </button>
          <button className="px-6 btn btn-grey rounded-md md:ml-4">
            <a href="https://github.com/stanford-crfm/helm">Github</a>
          </button>
        </div>
      </div>

      <div className="mx-4 mt-6 md:mt-0 md:w-1/3">
        <img
          src={helmHero}
          alt="HELM Hero"
          className="object-cover w-full h-full"
        />
      </div>
    </div>
  );
}
