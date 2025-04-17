import type Taxonomy from "@/types/Taxonomy";

export default interface GroupMetadata {
  description: string;
  display_name: string;
  taxonomy: Taxonomy | null;
}
