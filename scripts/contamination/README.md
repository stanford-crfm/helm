# Apache Beam implementation of HELM Contamination

## Installation

```bash
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3 venv

# Activate the virtual environment.
source venv/bin/activate

# Install requirements
pip install git+https://github.com/stanford-crfm/helm.git@67033775cae93ab06ece65d006091f72cf841826
pip install apache-beam~=2.46.0
pip install bitarray~=2.7.3
```

## Running

```bash
python3 run_contamination_beam.py --scenario-data /path/to/scenario_data.jsonl --input-data /path/to/input_data.jsonl --input-format the_pile --output-stats /path/to/output_stats.jsonl --normalization default --tags tag1 tag2
```

## API

Model developers should implement an Apache Beam pipeline that creates a `PCollection[str]` of documents, and then pass it to `ComputeAndWriteContaminationStats()` with the appropriate arguments.

Note: Each record in the `PCollection[str]` should contain an _entire_ document, not a single line from a document.

```python
with beam.Pipeline() as pipeline:
    _ = (
        pipeline
        # The model developer should modify these lines to read from the actual training set.
        | "Read" >> beam.io.ReadFromText(input_data)
        | "ExtractTextFromDocument" >> beam.Map(extract_text_from_document)
        # Call the HELM Contamination Apache Beam API.
        | "ComputeAndWriteContaminationStats" >> ComputeAndWriteContaminationStats(
            scenario_data_path=scenario_data,
            n_values=n_values,
            normalization=normalization,
            tags=tags
        )
    )
```

## Notes

The beam script does not support outputting contaminated ngrams yet.