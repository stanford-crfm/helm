import MediaObject from "@/types/MediaObject";

interface Props {
  mediaObject: MediaObject;
}

export default function MediaObjectDisplay({ mediaObject }: Props) {
  // TODO: Actually render the object.
  // NOTE: If mediaObject.location has the prefix "benchmark_output/",
  // this prefix should be stripped before appending the location to
  // `window.BENCHMARK_OUTPUT_BASE_URL`.
  return (
    <div>
      content_type: {mediaObject.content_type}, text:{" "}
      {mediaObject.text || "undefined"}, location:{" "}
      {mediaObject.location || "undefined"}
    </div>
  );
}
