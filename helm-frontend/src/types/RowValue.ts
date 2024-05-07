export default interface RowValue {
  href: string;
  markdown: boolean;
  value: string | number;
  run_spec_names?: string[];
  style?: Record<string, string>;
}
