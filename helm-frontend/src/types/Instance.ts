import type Reference from "@/types/Reference";

export default interface Instance {
  id: string;
  split: string;
  input: {
    text: string;
  };
  references: Reference[];
}
