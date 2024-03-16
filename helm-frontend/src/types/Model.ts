export type AccessLevel = "open" | "limited" | "closed";

export default interface Model {
  name: string;
  display_name: string;
  description: string;
  description_html: string;
  access: AccessLevel;
  creator_organization: string;
  num_parameters: string;
  release_date: Date;
  todo?: boolean;
}
