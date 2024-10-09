import MediaObject from "./MediaObject";

export default interface DisplayPrediction {
  media_object?: MediaObject;
  width?: number;
  height?: number;
  latex_code?: string;
  text?: string;
  html?: string;
  error?: string;
}
