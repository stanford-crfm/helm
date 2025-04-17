export default interface MetricField {
  name: string;
  display_name: string | undefined;
  short_display_name: string | undefined;
  description: string | undefined;
  lower_is_better: boolean | undefined;
}
