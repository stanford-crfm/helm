from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List

from helm.benchmark.presentation.schema import read_schema, SCHEMA_CLASSIC_YAML_FILENAME, ModelField
from helm.benchmark.run_expander import RUN_EXPANDERS
from helm.proxy.models import ALL_MODELS, Model


@dataclass(frozen=True)
class ModelInfo(ModelField):
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @staticmethod
    def from_model_field_and_model_object(model_field: ModelField, model_object: Optional[Model] = None):
        # Copy all attributes from model_field
        # and set group to model_field.creator_organization
        # and tags to an empty list
        if model_object is not None:
            model_info = ModelInfo(**model_field.__dict__, group=model_object.group, tags=model_object.tags)
        else:
            model_info = ModelInfo(**model_field.__dict__, group=model_field.creator_organization, tags=[])
        return model_info


def define_env(env):
    @env.macro
    def models_by_organization():
        # TODO: make this customizable
        schema = read_schema(SCHEMA_CLASSIC_YAML_FILENAME)
        result = defaultdict(list)

        # Create dict name -> madel_object (ALL_MODELS)
        name_to_model_object = {}
        for model_object in ALL_MODELS:
            name_to_model_object[model_object.name] = model_object

        for model_field in schema.models:
            model_object = name_to_model_object.get(model_field.name, None)
            model_info: ModelInfo = ModelInfo.from_model_field_and_model_object(model_field, model_object)
            result[model_info.creator_organization].append(model_info)
        if "Simple" in result:
            del result["Simple"]
        return result

    @env.macro
    def run_expanders():
        return RUN_EXPANDERS

    @env.macro
    def render_model_tags(model):
        return ", ".join([f"`{tag}`" for tag in model.tags])
