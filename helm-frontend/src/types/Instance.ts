import MultimediaObject from "@/types/MultimediaObject";
import type Reference from "@/types/Reference";

export default interface Instance {
  id: string;
  split: string;
  input: {
    text: string;
    multimedia_content: MultimediaObject | undefined;
  };
  references: Reference[];
}
