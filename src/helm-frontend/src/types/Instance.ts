interface Reference {
  output: {
    text: string;
  };
  tags: string[];
}

export default interface Instance {
  id: string;
  split: string;
  input: {
    text: string;
  };
  references: Reference[];
}
