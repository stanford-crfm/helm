import { useEffect, useState } from "react";
import {
  Card,
  CategoryBar,
  DonutChart,
  Legend,
  Metric,
  Text,
} from "@tremor/react";
import type Model from "@/types/Model";
import getSchema from "@/services/getSchema";
import PageTitle from "@/components/PageTitle";
import AccessBadge from "@/components/AccessBadge";
import MarkdownValue from "@/components/MarkdownValue";
import Loading from "@/components/Loading";

export default function Models() {
  const [models, setModels] = useState<Model[]>([] as Model[]);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const schema = await getSchema(controller.signal);
      setModels(schema.models);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  const [open, limited, closed] = models.reduce(
    (acc, model) => {
      switch (model.access) {
        case "open":
          acc[0] += 1;
          break;
        case "limited":
          acc[1] += 1;
          break;
        case "closed":
          acc[2] += 1;
          break;
      }
      return acc;
    },
    [0, 0, 0],
  );

  const creators = Object.values(
    models.reduce(
      (acc, model) => {
        const creator = model.creator_organization;
        if (acc[creator] === undefined) {
          acc[creator] = {
            name: creator,
            models: 1,
          };
          return acc;
        }

        acc[creator].models += 1;
        return acc;
      },
      {} as Record<string, { name: string; models: number }>,
    ),
  );

  if (models.length === 0) {
    return <Loading />;
  }

  return (
    <>
      <PageTitle title="Models" />

      <div className="overflow-x-auto mt-12">
        <table className="table">
          <thead>
            <tr>
              <th>Creator</th>
              <th>Model</th>
              <th>Description</th>
              <th>Access</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr>
                <td className="text-lg">{model.creator_organization}</td>
                <td>
                  <span className="text-xl">{model.display_name}</span>
                  <br />
                  <span>{model.name}</span>
                </td>
                <td>
                  <MarkdownValue value={model.description} />
                </td>
                <td>
                  <AccessBadge level={model.access} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <PageTitle title="Analysis" />
        <div className="grid md:grid-cols-3 grid-cols-1 gap-8">
          <Card className="flex flex-col justify-between">
            <Text>Models</Text>
            <Metric className="text-6xl md:!text-[96px]">
              {models.length}
            </Metric>
            <CategoryBar
              values={[open, limited, closed]}
              colors={["green", "yellow", "red"]}
            />
            <Legend
              categories={["Open", "Limited", "Closed"]}
              colors={["green", "yellow", "red"]}
            />
          </Card>
          <Card className="md:col-span-2">
            <Text>Creator Organizations</Text>
            <div className="flex justify-between mt-4">
              <DonutChart
                data={creators}
                category="models"
                index="name"
                variant="pie"
                className="basis-5/12"
              />
              <Legend
                categories={creators.map((creator) => creator.name)}
                className="basis-7/12"
              />
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
