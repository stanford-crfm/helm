import type LinkValue from "@/types/LinkValue";
import type RowValue from "@/types/RowValue";
import type HeaderValue from "@/types/HeaderValue";

export default interface GroupsTable {
  title: string;
  links: LinkValue[];
  header: HeaderValue[];
  rows: RowValue[][];
  name: string | undefined;
}
