export default interface Scenario {
  name: string;
  description: string;
  tags: string[];
  output_path: string;
  definition_path: string;
}
