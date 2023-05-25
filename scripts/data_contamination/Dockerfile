FROM python:3

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install -r ./requirements.txt

ENV INPUT_PATH=
ENV SCENARIO_PATH=
ENV OUTPUT_PATH=
ENV INPUT_FORMAT=

CMD python compute_contamination_metrics.py --input-data $INPUT_PATH --scenario-data $SCENARIO_DATA --output-stats $OUTPUT_STATS --input-format $INPUT_FORMAT

