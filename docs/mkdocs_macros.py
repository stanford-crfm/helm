from typing import Dict, List, Type

from helm.benchmark.model_metadata_registry import ALL_MODELS_METADATA, TEXT_MODEL_TAG, CODE_MODEL_TAG, DEPRECATED_MODEL_TAG, ModelMetadata
from helm.benchmark.run_expander import RUN_EXPANDERS, RunExpander


def define_env(env):

    @env.macro
    def models_by_organization_with_tag(tag: str) -> Dict[str, List[ModelMetadata]]:
        result: Dict[str, List[ModelMetadata]] = {}

        for model_metadata in ALL_MODELS_METADATA:
            if DEPRECATED_MODEL_TAG in model_metadata.tags:
                continue
            if tag not in model_metadata.tags:
                continue
            if model_metadata.creator_organization == "simple":
                continue
            creator_organization_name = model_metadata.creator_organization_name
            if creator_organization_name not in result:
                result[creator_organization_name] = []
            result[creator_organization_name].append(model_metadata)

        return result

    @env.macro
    def run_expanders() -> Dict[str, Type[RunExpander]]:
        return RUN_EXPANDERS
