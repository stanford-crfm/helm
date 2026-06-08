import json
from dataclasses import asdict
from datetime import date

from dacite import from_dict

from helm.benchmark.model_metadata_registry import (
    DACITE_CONFIG,
    ModelMetadata,
    TEXT_MODEL_TAG,
)
from helm.common.general import serialize_dates


def test_model_metadata_json_round_trip():
    """ModelMetadata has a `release_date` date field that the json standard
    library cannot round-trip on its own. Check that serialize_dates and the
    dacite type hook in DACITE_CONFIG together preserve the value."""
    model_metadata = ModelMetadata(
        name="openai/davinci",
        creator_organization_name="OpenAI",
        display_name="davinci",
        description="A model.",
        access="limited",
        release_date=date(2021, 8, 11),
        tags=[TEXT_MODEL_TAG],
    )

    serialized = json.dumps(asdict(model_metadata), default=serialize_dates)
    deserialized = from_dict(ModelMetadata, json.loads(serialized), config=DACITE_CONFIG)

    assert deserialized == model_metadata
    assert deserialized.release_date == date(2021, 8, 11)
