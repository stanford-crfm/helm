export default interface HeaderValue {
  value: string;
  markdown: boolean;
  description?: string;
  lower_is_better?: boolean;
  metadata: {
    [key: string]: unknown;
  };
}
