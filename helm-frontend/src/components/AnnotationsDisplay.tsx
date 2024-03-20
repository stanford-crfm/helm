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
              <h3>
                <strong>{key}</strong>
              </h3>
              {value.map((annotation, idx) => (
                <div key={idx}>
                  {annotation.error && (
                    <div>
                      <h3 className="ml-1">Error</h3>
                      <Preview value={annotation["error"]} />{" "}
                    </div>
                  )}
                  {annotation.text && (
                    <div>
                      <h3 className="ml-1">Text</h3>
                      <Preview value={annotation["text"]} />{" "}
                    </div>
                  )}
                  {annotation.media_object && (
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
