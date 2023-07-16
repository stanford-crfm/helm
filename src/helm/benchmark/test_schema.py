import os
import pytest
import yaml
import json
from jsonschema import validate, FormatChecker
from jsonschema.exceptions import ValidationError


class TestSchema:
    def setup_method(self, method):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "static", "schema.yaml"), "r") as stream:
            # This is to remove the YAML conversion to a Python date object
            self.instance = json.loads(json.dumps(yaml.safe_load(stream), default=str))
        with open(os.path.join(dir_path, "schema_schema.json"), "r") as stream:
            self.schema = json.load(stream)

    def test_models(self):
        try:
            validate(instance=self.instance, schema=self.schema, format_checker=FormatChecker())
        except ValidationError as err:
            pytest.fail(f"{err}")
