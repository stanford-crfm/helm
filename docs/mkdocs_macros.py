from collections import defaultdict

from helm.benchmark.run_expander import RUN_EXPANDERS
from helm.proxy.models import ALL_MODELS
import yaml


def define_env(env):
    @env.macro
    def models_by_organization():
        with open("src/helm/benchmark/static/schema.yaml", "r") as f:
            schema = yaml.safe_load(f)
        models = list(schema["models"])
        result = defaultdict(list)

        # Create dict name -> madel_object (ALL_MODELS)
        name_to_model_object = {}
        for model_object in ALL_MODELS:
            name_to_model_object[model_object.name] = model_object

        for model in models:
            # Find the model in ALL_MODELS in an efficient way
            model_object = name_to_model_object.get(model["name"], None)
            if model_object is not None:
                model["tags"] = model_object.tags
                model["group"] = model_object.group
            else:
                model["tags"] = []
                model["group"] = model["creator_organization"]
            result[model["creator_organization"]].append(model)
        if "Simple" in result:
            del result["Simple"]
        return result

    @env.macro
    def run_expanders():
        return RUN_EXPANDERS

    @env.macro
    def render_model_tags(model):
        return ", ".join([f"`{tag}`" for tag in model["tags"]])
