import MultimediaObject from "@/types/MultimediaObject";
import MediaObjectDisplay from "./MediaObjectDisplay";

interface Props {
  multimediaObject: MultimediaObject;
}

export default function MultimediaObjectDisplay({ multimediaObject }: Props) {
  return (
    <div>
      {multimediaObject.media_objects.map((mediaObject) => (
        <MediaObjectDisplay mediaObject={mediaObject} />
      ))}
    </div>
  );
}
