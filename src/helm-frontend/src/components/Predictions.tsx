import { useState } from "react";
import type DisplayPrediction from "@/types/DisplayPrediction";
import type DisplayRequest from "@/types/DisplayRequest";
import Indicator from "@/components/Indicator";
import Request from "@/components/Request";
import { List, ListItem } from "@tremor/react";
import Tab from "@/components/Tab";
import Tabs from "@/components/Tabs";
import PreView from "@/components/PreView";

type Props = {
  predictions: DisplayPrediction[];
  requests: DisplayRequest[];
};

/**
 * @SEE https://github.com/stanford-crfm/helm/blob/cffe38eb2c814d054c778064859b6e1551e5e106/src/helm/benchmark/static/benchmarking.js#L583-L679
 */
export default function Predictions({ predictions, requests }: Props) {
  const [openRequests, setOpenRequests] = useState(false);
  const [openDetails, setOpenDetails] = useState(true);
  const handleToggle = () => {
    setOpenRequests(!openRequests);
    setOpenDetails(!openDetails);
  };

  if (predictions.length < 1) {
    return null;
  }

  return (
    <div>
      <div className="flex flex-wrap justify-start items-start">
        {predictions.map((prediction, idx) => (
          <div className="w-full" key={idx}>
            <div className="mt-2 w-full">
              <h3>
                <span className="mr-4">
                  {`Prediction [trial ${prediction.train_trial_index}]`}
                </span>
                <Indicator stats={prediction.stats} />
              </h3>
              <PreView value={prediction.predicted_text} />
            </div>
            <div className="my-4">
              <Tabs>
                <Tab
                  onClick={handleToggle}
                  active={openDetails}
                >
                  Details
                </Tab>
                <Tab
                  onClick={handleToggle}
                  active={openRequests}
                >
                  Requests
                </Tab>
              </Tabs>
            </div>
            {openDetails
              ? (
                <List>
                  {(Object.keys(
                    prediction.stats,
                  ) as (keyof typeof prediction.stats)[]).map((
                    statKey,
                    idx,
                  ) => (
                    <ListItem key={idx}>
                      <span>{statKey}:</span>
                      <span>{prediction.stats[statKey]}</span>
                    </ListItem>
                  ))}
                </List>
              )
              : null}
            <div className="overflow-auto">
              {openRequests ? <Request request={requests[idx]} /> : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
