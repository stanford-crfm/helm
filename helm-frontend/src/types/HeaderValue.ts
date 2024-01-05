export default interface HeaderValue {
  value: string;
  markdown: boolean;
  description?: string;
  metadata: {
    [key: string]: unknown;
  };
}
