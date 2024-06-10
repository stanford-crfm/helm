import CompletionAnnotation from "@/types/CompletionAnnotation";
import Preview from "./Preview";
import MediaObjectDisplay from "./MediaObjectDisplay";

// TODO: This is a dirty hack to support annotations from
// Image2Structure and AIRBench, but eventually we should make sure
// all annotations are supported generally.
type Props = {
  predictionAnnotations:
    | Record<
        string,
        Array<CompletionAnnotation> | Record<string, string | number>
      >
    | undefined;
};

function listAnnotationDisplay(listAnnotation: Array<CompletionAnnotation>) {
  return (
    <div>
      {listAnnotation.map((annotation, idx) => (
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
            <MediaObjectDisplay mediaObject={annotation["media_object"]} />
          )}
        </div>
      ))}
    </div>
  );
}

function dictAnnotationDisplay(
  dictAnnotation: Record<string, string | number>,
) {
  return (
    <div>
      {Object.entries(dictAnnotation).map(([key, value]) => (
        <div>
          <h3 className="ml-1">{key}</h3>
          <Preview value={value.toString()} />
        </div>
      ))}
    </div>
  );
}

export default function AnnotationDisplay({ predictionAnnotations }: Props) {
  return (
    <div>
      {predictionAnnotations && predictionAnnotations !== undefined
        ? Object.entries(predictionAnnotations).map(([key, value]) => (
            <div key={key}>
              <h3>
                <strong>{key}</strong>
              </h3>
              {Array.isArray(value)
                ? listAnnotationDisplay(value)
                : dictAnnotationDisplay(value)}
            </div>
          ))
        : null}
    </div>
  );
}
