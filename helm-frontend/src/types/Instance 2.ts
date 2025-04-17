import MultimediaObject from "@/types/MultimediaObject";
import type Reference from "@/types/Reference";
import Perturbation from "@/types/Perturbation";
import Message from "@/types/Message";

export default interface Instance {
  id: string;
  split: string;
  input: {
    text: string;
    multimedia_content: MultimediaObject | undefined;
    messages: Message[] | undefined;
  };
  references: Reference[];
  perturbation?: Perturbation | undefined;
  extra_data?: Record<
    string,
    string | number | null | Array<string | number | null>
  >;
}
