import type Instance from "@/types/Instance";
import type DisplayRequest from "@/types/DisplayRequest";
import type DisplayPrediction from "@/types/DisplayPrediction";
import Predictions from "@/components/Predictions";
import References from "@/components/References";
import type MetricFieldMap from "@/types/MetricFieldMap";
import Preview from "@/components/Preview";

interface Props {
  instance: Instance;
  requests: DisplayRequest[];
  predictions: DisplayPrediction[];
  metricFieldMap: MetricFieldMap;
}

export default function InstanceData({
  instance,
  requests,
  predictions,
  metricFieldMap,
}: Props) {
  return (
    <div className="border p-4">
      <div>
        <h3 className="text-xl mb-4">
          {`Instance id: ${instance.id} [split:  ${instance.split}]`}
        </h3>
        <h3>Input</h3>
        <Preview value={instance.input.text} />

        <div>
          {instance.references && instance.references.length > 0 ? (
            <References references={instance.references} />
          ) : null}
        </div>
        <div>
          {predictions && requests ? (
            <Predictions
              predictions={predictions}
              requests={requests}
              metricFieldMap={metricFieldMap}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}
