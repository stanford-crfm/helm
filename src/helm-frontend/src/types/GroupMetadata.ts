import type Taxonomy from "@/types/Taxonomy";

export default interface GroupMetaData {
  description: string;
  display_name: string;
  taxonomy: Taxonomy | null;
}
