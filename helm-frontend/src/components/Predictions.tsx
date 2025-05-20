import type DisplayPrediction from "@/types/DisplayPrediction";
import type DisplayRequest from "@/types/DisplayRequest";
import type MetricFieldMap from "@/types/MetricFieldMap";
import Indicator from "@/components/Indicator";
import Request from "@/components/Request";
import Preview from "@/components/Preview";
import { List, ListItem } from "@tremor/react";
import AnnotationsDisplay from "./AnnotationsDisplay";

type Props = {
  predictions: DisplayPrediction[];
  requests: DisplayRequest[];
  metricFieldMap: MetricFieldMap;
};

/**
 * @SEE https://github.com/stanford-crfm/helm/blob/cffe38eb2c814d054c778064859b6e1551e5e106/src/helm/benchmark/static/benchmarking.js#L583-L679
 */
export default function Predictions({
  predictions,
  requests,
  metricFieldMap,
}: Props) {
  if (predictions.length < 1) {
    return null;
  }

  return (
    <div>
      <div className="flex flex-wrap justify-start items-start">
        {predictions.map((prediction, idx) => (
          <div className="w-full" key={idx}>
            {predictions.length > 1 ? (
              <h2>Trial {prediction.train_trial_index}</h2>
            ) : null}
            <div className="mt-2 w-full">
              {prediction.thinking_text ? (
                <>
                  <h3>
                    <span className="mr-4">Thinking</span>
                  </h3>
                  <Preview value={prediction.thinking_text} />
                </>
              ) : null}
              {prediction.base64_images &&
              prediction.base64_images.length > 0 ? (
                <>
                  <h3 className="mr-4">Prediction image</h3>
                  {prediction.base64_images.map((base64_image) => (
                    <img
                      src={"data:image;base64," + base64_image}
                      alt="Base64 Image"
                    />
                  ))}
                </>
              ) : (
                <>
                  <h3>
                    <span className="mr-4">Prediction raw text</span>
                    <Indicator stats={prediction.stats} />
                  </h3>
                  <Preview value={prediction.predicted_text} />
                  {prediction.mapped_output ? (
                    <>
                      <h3>Prediction mapped output</h3>
                      <Preview value={String(prediction.mapped_output)} />
                    </>
                  ) : null}
                </>
              )}
            </div>
            <AnnotationsDisplay
              predictionAnnotations={prediction.annotations}
            />
            <div className="mx-1">
              <h3>Metrics</h3>
              <List>
                {Object.keys(prediction.stats).map((statKey, idx) => (
                  <ListItem key={idx}>
                    {metricFieldMap[statKey] ? (
                      <span title={metricFieldMap[statKey].description}>
                        {metricFieldMap[statKey].display_name}
                      </span>
                    ) : (
                      <span>{statKey}</span>
                    )}
                    <span>{String(prediction.stats[statKey])}</span>
                  </ListItem>
                ))}
              </List>
            </div>
            <details className="collapse collapse-arrow border rounded-md bg-white">
              <summary className="collapse-title">Request details</summary>
              <div className="collapse-content">
                <Request request={requests[idx]} />
              </div>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
}
