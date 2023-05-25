FROM python:3

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install -r ./requirements.txt

CMD python compute_contamination_metrics.py --input-data $INPUT_PATH --scenario-data $SCENARIO_DATA --output-stats $OUTPUT_STATS --input-format $INPUT_FORMAT

