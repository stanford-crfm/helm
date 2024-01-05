export default interface MetricGroup {
  name: string;
  display_name: string;
  metrics: {
    name: string;
    split: string;
  }[];
}
