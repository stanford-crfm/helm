import MultimediaObject from "@/types/MultimediaObject";
import type Reference from "@/types/Reference";
import Perturbation from "./Perturbation";

export default interface Instance {
  id: string;
  split: string;
  input: {
    text: string;
    multimedia_content: MultimediaObject | undefined;
  };
  references: Reference[];
  perturbation?: Perturbation | undefined;
  extra_data?:
    | Record<
        string,
        Array<string | number | null> | Record<string, string | number | null>
      >
    | undefined;
}
