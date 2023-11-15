import type Model from "@/types/Model";
import { Link as ReactRouterLink } from "react-router-dom";

interface Props {
  models: Model[];
}

export default function ModelsList({ models }: Props) {
  return (
    <section>
      <h3 className="text-3xl">{models.length} models</h3>
      <ul>
        {models.map((model, idx) =>
          model.todo ? (
            <li key={idx} className="text-slate-300 mt-1">
              {model.creator_organization} / {model.display_name}
            </li>
          ) : (
            <ReactRouterLink to={"models"}>
              <li className="text-black mt-1" key={idx}>
                {model.creator_organization} / {model.display_name}
              </li>
            </ReactRouterLink>
          ),
        )}
      </ul>
    </section>
  );
}
