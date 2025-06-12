import type Instance from "@/types/Instance";
import type DisplayRequest from "@/types/DisplayRequest";
import type DisplayPrediction from "@/types/DisplayPrediction";
import Predictions from "@/components/Predictions";
import References from "@/components/References";
import type MetricFieldMap from "@/types/MetricFieldMap";
import Preview from "@/components/Preview";
import MultimediaObjectDisplay from "@/components/MultimediaObjectDisplay";
import ExtraDataDisplay from "@/components/ExtraDataDisplay";
import MessagesDisplay from "@/components/MessagesDisplay";

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
    <div>
      <h3>Input</h3>
      {instance.input.multimedia_content !== undefined ? (
        <MultimediaObjectDisplay
          multimediaObject={instance.input.multimedia_content}
        />
      ) : instance.input.text.includes(`<br><img src="data:image;base64`) ? (
        <div dangerouslySetInnerHTML={{ __html: instance.input.text }} />
      ) : instance.input.messages !== undefined &&
        instance.input.messages.length ? (
        <MessagesDisplay messages={instance.input.messages} />
      ) : (
        <Preview value={instance.input.text} />
      )}

      <div>
        {instance.references && instance.references.length > 0 ? (
          <References references={instance.references} />
        ) : null}
      </div>
      {instance.extra_data && instance.extra_data.length ? (
        <ExtraDataDisplay extraData={instance.extra_data} />
      ) : null}
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
  );
}
