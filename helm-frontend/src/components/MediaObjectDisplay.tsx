import MediaObject from "@/types/MediaObject";

interface Props {
  mediaObject: MediaObject;
}

export default function MediaObjectDisplay({ mediaObject }: Props) {
  return (
    <div>
      content_type: {mediaObject.content_type}, text:{" "}
      {mediaObject.text || "undefined"}, location:{" "}
      {mediaObject.location || "undefined"}
    </div>
  );
}
