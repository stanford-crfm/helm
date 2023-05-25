FROM python:3.8

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install -r ./requirements-freeze.txt

ENTRYPOINT python compute_contamination_metrics.py 
