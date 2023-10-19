export default interface Adapter {
  name: string;
  description: string;
  values?: {
    name: string;
    description: string;
  }[];
}
