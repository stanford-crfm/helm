import CompletionAnnotation from "@/types/CompletionAnnotation";
import Preview from "./Preview";
import MediaObjectDisplay from "./MediaObjectDisplay";

type Props = {
  predictionAnnotations:
    | Record<
        string,
        Array<CompletionAnnotation> | Record<string, string | number>
      >
    | undefined;
};

function listAnnotationDisplay(listAnnotation: Array<CompletionAnnotation>) {
  // This is a dirty hack to support annotations from Image2Struct.
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

function annotationToEntries(
  annotation: unknown,
  keyPrefix?: string | undefined,
): Array<[string, string]> {
  if (Array.isArray(annotation)) {
    return annotation.flatMap((item, index) =>
      annotationToEntries(item, `${keyPrefix || ""}[${index}]`),
    );
  } else if (
    annotation instanceof Object &&
    annotation.constructor === Object
  ) {
    return Object.entries(annotation).flatMap(([key, value]) =>
      annotationToEntries(value, keyPrefix ? `${keyPrefix}.${key}` : key),
    );
  } else {
    return [
      [
        keyPrefix || "",
        typeof annotation === "string"
          ? annotation
          : JSON.stringify(annotation),
      ],
    ];
  }
}

function dictAnnotationDisplay(dictAnnotation: Record<string, unknown>) {
  return (
    <div>
      {annotationToEntries(dictAnnotation).map(([key, value]) => (
        <div key={key}>
          <h3 className="ml-1">{key}</h3>
          <Preview value={value} />
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
            <details
              key={key}
              className="collapse collapse-arrow border rounded-md bg-white my-2"
            >
              <summary className="collapse-title">
                <>{"View " + key + " annotations"}</>
              </summary>
              <div className="collapse-content">
                {Array.isArray(value)
                  ? listAnnotationDisplay(value)
                  : dictAnnotationDisplay(value)}
              </div>
            </details>
          ))
        : null}
    </div>
  );
}
