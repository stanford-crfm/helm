import CompletionAnnotation from "@/types/CompletionAnnotation";
import Preview from "./Preview";

type Props = {
  predictionAnnotations:
    | Record<string, Array<CompletionAnnotation>>
    | undefined;
};

export default function AnnotationDsisplay({ predictionAnnotations }: Props) {
  return (
    <div>
      {predictionAnnotations && predictionAnnotations !== undefined
        ? Object.entries(predictionAnnotations).map(([key, value]) => (
            <div key={key}>
              <h3>{key}</h3>
              <Preview value={JSON.stringify(value[0])} />
            </div>
          ))
        : null}
    </div>
  );
}
