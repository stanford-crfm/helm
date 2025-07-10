import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Card,
  List,
  ListItem,
  Text,
  Metric,
  Flex,
  ProgressBar,
  Title,
  Badge,
} from "@tremor/react";
import type LLMJudgeData from "@/types/LLMJudgeData";
import type { LLMJudgeTask } from "@/types/LLMJudgeData";
import Loading from "@/components/Loading";
import Pagination from "@/components/Pagination";
import Preview from "@/components/Preview";

const TASKS_PAGE_SIZE = 10;

type Props = {
  data: LLMJudgeData | undefined;
  isLoading: boolean;
};

export default function LLMJudgeDisplay({ data, isLoading }: Props) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [currentTasksPage, setCurrentTasksPage] = useState<number>(() => {
    const pageFromUrl = searchParams.get("tasksPage");
    return pageFromUrl ? parseInt(pageFromUrl, 10) : 1;
  });
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("tasksPage", String(currentTasksPage));
    setSearchParams(newSearchParams, { replace: true });
  }, [currentTasksPage, setSearchParams, searchParams]);

  if (isLoading) {
    return (
      <Card>
        <div className="flex justify-center items-center h-40">
          <Loading />
        </div>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <Title>LLM Judge Evaluation</Title>
        <Text className="mt-2">
          No LLM Judge evaluation available for this run.
        </Text>
        <Text className="mt-1">
          The file <code>llm_judge_summary.json</code> was not found or could
          not be loaded.
        </Text>
      </Card>
    );
  }

  const filteredTasks = data.tasks.filter(
    (task) =>
      task.explanation.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (task.instance_id &&
        String(task.instance_id)
          .toLowerCase()
          .includes(searchTerm.toLowerCase())) ||
      String(task.judgement).toLowerCase().includes(searchTerm.toLowerCase()),
  );

  const totalTasksPages = Math.ceil(filteredTasks.length / TASKS_PAGE_SIZE);
  const pagedTasks = filteredTasks.slice(
    (currentTasksPage - 1) * TASKS_PAGE_SIZE,
    currentTasksPage * TASKS_PAGE_SIZE,
  );

  const handleNextPage = () => {
    setCurrentTasksPage((prev) => Math.min(prev + 1, totalTasksPages));
  };

  const handlePrevPage = () => {
    setCurrentTasksPage((prev) => Math.max(prev - 1, 1));
  };

  return (
    <div className="space-y-6">
      <Card>
        <Title>LLM Judge Evaluation Summary</Title>
        <List className="mt-4">
          <ListItem>
            <Text>Evaluated Model:</Text>
            <Text className="font-medium">{data.main_model}</Text>
          </ListItem>
          <ListItem>
            <Text>Judge Model:</Text>
            <Text className="font-medium">{data.judge_model}</Text>
          </ListItem>
          <ListItem>
            <Text>Agreement Level:</Text>
            <Flex
              justifyContent="end"
              alignItems="baseline"
              className="space-x-2"
            >
              <Metric>{(data.agreement_level * 100).toFixed(2)}%</Metric>
              <Text>
                ({data.agreements} / {data.total_valid_instances})
              </Text>
            </Flex>
          </ListItem>
          {data.total_valid_instances > 0 ? (
            <ProgressBar
              value={data.agreement_level * 100}
              color="teal"
              className="mt-1"
            />
          ) : null}
          <ListItem>
            <Text>Total Instances:</Text>
            <Text className="font-medium">{data.total_judged_instances}</Text>
          </ListItem>
          <ListItem>
            <Text>Total Valid Judged Instances:</Text>
            <Text className="font-medium">{data.total_valid_instances}</Text>
          </ListItem>
          {data.invalid_instances > 0 ? (
            <ListItem>
              <Text className="text-tremor-content-attention dark:text-dark-tremor-content-attention">
                Instances with Invalid/Malformed Judge Responses:
              </Text>
              <Text className="font-medium text-tremor-content-attention dark:text-dark-tremor-content-attention">
                {data.invalid_instances}
              </Text>
            </ListItem>
          ) : null}
        </List>
      </Card>

      {data.tasks.length > 0 ? (
        <Card>
          <Flex alignItems="baseline" justifyContent="between">
            <Title>Individual Evaluation Details</Title>
            <Text>{filteredTasks.length} tasks found</Text>
          </Flex>
          <div className="flex justify-start my-4">
            <input
              type="text"
              className="input input-bordered w-full max-w-xs"
              placeholder="Search explanations, IDs or judgements..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentTasksPage(1);
              }}
            />
          </div>

          {pagedTasks.length > 0 ? (
            <div className="space-y-4">
              {pagedTasks.map((task: LLMJudgeTask, index: number) => (
                <Card
                  key={
                    task.instance_id != null
                      ? String(task.instance_id)
                      : `task-${index}`
                  }
                  className="p-4 ring-1 ring-gray-200"
                >
                  {task.instance_id != null ? (
                    <h3 className="text-xl mb-4">
                      Instance id: {String(task.instance_id)}
                    </h3>
                  ) : null}
                  <h3>
                    <span className="mr-4">Judgement:</span>
                    <span>
                      <Badge color={task.judgement === 1 ? "emerald" : "rose"}>
                        {task.judgement === 1 ? "Agreement" : "Disagreement"}
                      </Badge>
                    </span>
                  </h3>

                  <div className="mt-2">
                    <h3>
                      <span className="mr-4">Judge Explanation:</span>
                    </h3>
                    <Preview value={String(task.explanation)} />
                  </div>
                  {Object.entries(task).filter(
                    ([key]) =>
                      !["judgement", "explanation", "instance_id"].includes(
                        key,
                      ),
                  ).length > 0 ? (
                    <div className="mt-3">
                      <h3>
                        <span className="mr-4">Input:</span>
                      </h3>
                      <Preview value={String(task.input)} />
                      <h3>
                        <span className="mr-4">
                          Prediction of the Evaluated Model:
                        </span>
                      </h3>
                      <Preview value={String(task.prediction)} />
                    </div>
                  ) : null}
                </Card>
              ))}
            </div>
          ) : (
            <Text className="mt-4 text-center">
              No tasks found with the current search term.
            </Text>
          )}

          {totalTasksPages > 1 ? (
            <Pagination
              className="flex justify-center mt-8 mb-2"
              onNextPage={handleNextPage}
              onPrevPage={handlePrevPage}
              currentPage={currentTasksPage}
              totalPages={totalTasksPages}
            />
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
