FROM python:3

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install -r ./requirements.txt

ENV INPUT_PATH=input.json
ENV SCENARIO_PATH=scenario_data
ENV OUTPUT_PATH=output_stats
ENV INPUT_FORMAT=the_pile

CMD python compute_contamination_metrics.py --input-data $INPUT_PATH --scenario-data $SCENARIO_DATA --output-stats $OUTPUT_STATS --input-format $INPUT_FORMAT

