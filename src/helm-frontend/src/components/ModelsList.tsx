import type Model from "@/types/Model";

interface Props {
  models: Model[];
}

export default function ModelsList({ models }: Props) {
  return (
    <section>
      <h3 className="text-3xl">{models.length} Models</h3>
      <ul>
        {models.map((model, idx) => (
          <li key={idx}>
            {model.creator_organization} / {model.display_name}
          </li>
        ))}
      </ul>
    </section>
  );
}
