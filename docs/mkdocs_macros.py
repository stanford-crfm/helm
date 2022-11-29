from collections import defaultdict

from helm.benchmark.run_expander import RUN_EXPANDERS
from helm.proxy.models import ALL_MODELS


def define_env(env):
    @env.macro
    def models_by_organization():
        result = defaultdict(list)
        for model in ALL_MODELS:
            result[model.creator_organization].append(model)
        if "Simple" in result:
            del result["Simple"]
        return result

    @env.macro
    def run_expanders():
        return RUN_EXPANDERS

    @env.macro
    def render_model_tags(model):
        return ", ".join([f"`{tag}`" for tag in model.tags])
