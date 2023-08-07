import type Model from "@/types/Model";

interface Props {
  models: Model[];
}

export default function ModelsList({ models }: Props) {
  return (
    <ul>
      <h3 className="text-3xl">{models.length} Models</h3>
      {models.map((model, idx) =>
        model.todo
          ? (
            <li key={idx} className="text-slate-300">
              {model.creator_organization} / {model.display_name}
            </li>
          )
          : (
            <li key={idx}>
              {model.creator_organization} / {model.display_name}
            </li>
          )
      )}
    </ul>
  );
}
