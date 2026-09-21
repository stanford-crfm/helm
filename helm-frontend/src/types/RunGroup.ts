import type Taxonomy from "@/types/Taxonomy";
/**
 * @TODO This is likely combining distinct entities
 * We're just using optional properties because
 * we don't actually know what those are right now
 */
export default interface RunGroup {
  name: string;
  display_name: string;
  short_display_name?: string;
  description: string;
  environment?: {
    main_name?: string;
    main_split?: string;
  };
  metric_groups?: string[];
  taxonomy?: Taxonomy;
  todo?: boolean;
  category?: string;
  subgroups?: string[];
  visibility?: string;
  short_description?: string;
}
