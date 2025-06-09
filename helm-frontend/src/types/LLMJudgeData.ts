export interface LLMJudgeTask {
  judgement: number;
  explanation: string;
  [key: string]: unknown;
}

export default interface LLMJudgeData {
  benchmark: string;
  main_model: string;
  judge_model: string;
  agreement_level: number;
  agreements: number;
  total_valid_instances: number;
  total_judged_instances: number;
  invalid_instances: number;
  tasks: LLMJudgeTask[];
}
