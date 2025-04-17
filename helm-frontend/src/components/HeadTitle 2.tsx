interface Props {
  title: string;
}

export default function Title({ title }: Props) {
  return (
    <title>{`${title} - Holistic Evaluation of Language Models (HELM)`}</title>
  );
}
