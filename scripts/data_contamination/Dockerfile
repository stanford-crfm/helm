FROM python:3

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install -r ./requirements.txt

CMD python compute_contamination_metrics.py --input-data input.json --scenario-data scenario_data --output-stats output_stats --input-format the_pile

