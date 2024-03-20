import CompletionAnnotation from "@/types/CompletionAnnotation";
import Preview from "./Preview";
import MediaObjectDisplay from "./MediaObjectDisplay";

type Props = {
  predictionAnnotations:
    | Record<string, Array<CompletionAnnotation>>
    | undefined;
};

export default function AnnotationDisplay({ predictionAnnotations }: Props) {
  return (
    <div>
      {predictionAnnotations && predictionAnnotations !== undefined
        ? Object.entries(predictionAnnotations).map(([key, value]) => (
            <div key={key}>
              <h3>{key}</h3>
              {value.map((annotation, idx) => (
                <div key={idx}>
                  {annotation.error && <Preview value={annotation["error"]} />}
                  {annotation.text && <Preview value={annotation["text"]} />}
                  {/* TODO remove this impossible condition, once Yifan's PR is merged. */}
                  {annotation.media_object && !annotation.media_object && (
                    <MediaObjectDisplay
                      mediaObject={annotation["media_object"]}
                    />
                  )}
                </div>
              ))}
            </div>
          ))
        : null}
    </div>
  );
}
