import { useState } from "react";
import type DisplayPrediction from "@/types/DisplayPrediction";
import type DisplayRequest from "@/types/DisplayRequest";
import Indicator from "@/components/Indicator";
import Request from "@/components/Request";
import Preview from "@/components/Preview";
import { List, ListItem } from "@tremor/react";

type Props = {
  predictions: DisplayPrediction[];
  requests: DisplayRequest[];
};

/**
 * @SEE https://github.com/stanford-crfm/helm/blob/cffe38eb2c814d054c778064859b6e1551e5e106/src/helm/benchmark/static/benchmarking.js#L583-L679
 */
export default function Predictions({ predictions, requests }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  const toggleAccordion = () => {
    setIsOpen(!isOpen);
  };

  if (predictions.length < 1) {
    return null;
  }

  if (predictions && predictions[0] && predictions[0].base64_images) {
    return (
      <div>
        <div className="flex flex-wrap justify-start items-start">
          {predictions.map((prediction, idx) => (
            <div className="w-full" key={idx}>
              {predictions.length > 1 ? <h2>Trial {idx}</h2> : null}
              <div className="mt-2 w-full">
                <h3>
                  <span className="mr-4">Prediction image</span>
                </h3>
                <div>
                  {prediction &&
                  prediction.base64_images &&
                  prediction.base64_images[0] ? (
                    <img
                      src={"data:image;base64," + prediction.base64_images[0]}
                      alt="Base64 Image"
                    />
                  ) : null}
                </div>
              </div>
              <div className="accordion-wrapper">
                <button
                  className="accordion-title p-5 bg-gray-100 hover:bg-gray-200 w-full text-left"
                  onClick={toggleAccordion}
                >
                  <h3 className="text-lg font-medium text-gray-900">
                    Prompt Details
                  </h3>
                </button>

                {isOpen && (
                  <div className="accordion-content p-5 border shadow-lg rounded-md bg-white">
                    <div className="mt-3 text-left">
                      <div className="overflow-auto">
                        <Request request={requests[idx]} />
                      </div>
                      <List>
                        {Object.keys(prediction.stats).map((statKey, idx) => (
                          <ListItem key={idx} className="mt-2">
                            <span>{statKey}:</span>
                            <span>
                              {String(
                                prediction.stats[
                                  statKey as keyof typeof prediction.stats
                                ],
                              )}
                            </span>
                          </ListItem>
                        ))}
                      </List>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap justify-start items-start">
        {predictions.map((prediction, idx) => (
          <div className="w-full" key={idx}>
            {predictions.length > 1 ? <h2>Trial {idx}</h2> : null}
            <div className="mt-2 w-full">
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
            </div>
            <div className="accordion-wrapper">
              <button
                className="accordion-title p-5 bg-gray-100 hover:bg-gray-200 w-full text-left"
                onClick={toggleAccordion}
              >
                <h3 className="text-lg font-medium text-gray-900">
                  Prompt Details
                </h3>
              </button>

              {isOpen && (
                <div className="accordion-content p-5 border shadow-lg rounded-md bg-white">
                  <div className="mt-3 text-left">
                    <div className="overflow-auto">
                      <Request request={requests[idx]} />
                    </div>
                    <List>
                      {Object.keys(prediction.stats).map((statKey, idx) => (
                        <ListItem key={idx} className="mt-2">
                          <span>{statKey}:</span>
                          <span>
                            {String(
                              prediction.stats[
                                statKey as keyof typeof prediction.stats
                              ],
                            )}
                          </span>
                        </ListItem>
                      ))}
                    </List>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
