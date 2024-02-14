import type Instance from "@/types/Instance";
import type DisplayRequest from "@/types/DisplayRequest";
import type DisplayPrediction from "@/types/DisplayPrediction";
import Predictions from "@/components/Predictions";
import References from "@/components/References";
import Preview from "@/components/Preview";

interface Props {
  instance: Instance;
  requests: DisplayRequest[];
  predictions: DisplayPrediction[];
}

export default function InstanceData({
  instance,
  requests,
  predictions,
}: Props) {
  return (
    <div className="border p-4">
      <h3 className="text-xl mb-4">
        {`Instance id: ${instance.id} [split:  ${instance.split}]`}
      </h3>
      <h3>Input</h3>
      <Preview value={instance.input.text} />
      <h3>References</h3>
      <span>
        <References references={instance.references} />
      </span>
      {predictions && requests ? (
        <Predictions predictions={predictions} requests={requests} />
      ) : null}
    </div>
  );
}
