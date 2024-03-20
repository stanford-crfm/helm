import MediaObject from "@/types/MediaObject";
import getBenchmarkEndpoint from "@/utils/getBenchmarkEndpoint";

interface Props {
  mediaObject: MediaObject;
}

export default function MediaObjectDisplay({ mediaObject }: Props) {
  if (mediaObject.content_type.includes("image")) {
    if (mediaObject.location === undefined) {
      return null;
    }
    const url = getBenchmarkEndpoint(
      mediaObject.location
        .replace("benchmark_output/", "")
        .replace("prod_env/", "../"),
    );
    return (
      <div>
        <img src={url}></img>
        <br />
      </div>
    );
  } else {
    if (
      mediaObject.text &&
      mediaObject.content_type &&
      mediaObject.content_type === "text/plain" &&
      mediaObject.text.length > 1
    ) {
      return (
        <div>
          {mediaObject.text}
          <br />
          <br />
        </div>
      );
    } else {
      return <div></div>;
    }
  }
}
