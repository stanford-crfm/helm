FROM python:3.8

RUN mkdir -p /script/src

WORKDIR /script/src

COPY . .

RUN pip install -r ./requirements-freeze.txt

ENTRYPOINT ["python", "compute_data_overlap_metrics.py"]

