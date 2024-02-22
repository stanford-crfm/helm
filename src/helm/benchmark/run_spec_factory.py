import dataclasses
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.model_deployment_registry import (
    ALL_MODEL_DEPLOYMENTS,
    DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT,
    ModelDeployment,
    get_model_deployment,
)
from helm.benchmark.model_metadata_registry import (
    ANTHROPIC_CLAUDE_1_MODEL_TAG,
    ANTHROPIC_CLAUDE_2_MODEL_TAG,
    BUGGY_TEMP_0_TAG,
    CHATML_MODEL_TAG,
    GOOGLE_GEMINI_MODEL_TAG,
    GOOGLE_PALM_2_MODEL_TAG,
    IDEFICS_INSTRUCT_MODEL_TAG,
    LLAVA_MODEL_TAG,
    NLG_PREFIX_TAG,
    NO_NEWLINES_TAG,
    OPENAI_CHATGPT_MODEL_TAG,
    ModelMetadata,
    get_model_metadata,
)
from helm.benchmark.run_expander import (
    RUN_EXPANDERS,
    AnthropicRunExpander,
    ChatMLRunExpander,
    GlobalPrefixRunExpander,
    GoogleRunExpander,
    IDEFICSInstructRunExpander,
    IncreaseTemperatureRunExpander,
    LlavaRunExpander,
    OpenAIRunExpander,
    StopRunExpander,
)
from helm.benchmark.run_spec import RunSpec, get_run_spec_function
from helm.common.general import singleton
from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import ObjectSpec


def get_default_model_deployment_for_model(
    model_name: str, warn_arg_deprecated: bool = False, ignore_deprecated: bool = False
) -> Optional[str]:
    """Returns a valid model deployment name corresponding to the given model arg.
    This is used as a backwards compatibility layer for model names that are now moved to model deployments.
    Example: "anthropic/claude-v1.3" => "anthropic/claude-v1.3"
    Example: "meta/llama-7b" => "together/llama-7b"

    The process to find a model deployment name is as follows:
    1. If there is a model deployment with the same name as the model arg, use it.
    2. If there is at least one deployment for the model, use the first one that is available.
    3. If there are no deployments for the model, returns None.

    This function will also try to find a model deployment name that is not deprecated.
    If there are no non-deprecated deployments, it will return the first deployment (even if it's deprecated).
    If ignore_deprecated is True, this function will return None if the model deployment is deprecated.

    If warn_arg_deprecated is True, this function will print a warning if the model deployment name is not the same
    as the model arg. This is to remind the user that the model name is deprecated and should be replaced with
    the model deployment name (in their config).

    Args:
        model_arg: The model arg to convert to a model deployment name.
        warn_arg_deprecated: Whether to print a warning if the model deployment name is not the same as the model arg.
        ignore_deprecated: Whether to return None if the model deployment is deprecated.
    """

    # If there is a model deployment with the same name as the model arg, use it.
    if model_name in DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT:
        deployment: ModelDeployment = DEPLOYMENT_NAME_TO_MODEL_DEPLOYMENT[model_name]
        if deployment.deprecated and ignore_deprecated:
            if warn_arg_deprecated:
                hlog(f"WARNING: Model deployment {model_name} is deprecated")
            return None
        return deployment.name

    # If there is at least one deployment for the model, use the first one that is available.
    available_deployments: List[ModelDeployment] = [
        deployment for deployment in ALL_MODEL_DEPLOYMENTS if deployment.model_name == model_name
    ]
    if len(available_deployments) > 0:
        available_deployment_names: List[str] = [deployment.name for deployment in available_deployments]
        if warn_arg_deprecated:
            hlog("WARNING: Model name is deprecated. Please use the model deployment name instead.")
            hlog(f"Available model deployments for model {model_name}: {available_deployment_names}")

        # Additionally, if there is a non-deprecated deployment, use it.
        non_deprecated_deployments: List[ModelDeployment] = [
            deployment for deployment in available_deployments if not deployment.deprecated
        ]
        if len(non_deprecated_deployments) > 0:
            chosen_deployment = non_deprecated_deployments[0]
        # There are no non-deprecated deployments, so there are two options:
        # 1. If we can return an empty string, return it. (no model deployment is available)
        # 2. If we can't return an empty string, return the first deployment (even if it's deprecated).
        elif ignore_deprecated:
            return None
        else:
            chosen_deployment = available_deployments[0]
            if warn_arg_deprecated:
                hlog(f"WARNING: All model deployments for model {model_name} are deprecated.")
        if warn_arg_deprecated:
            hlog(
                f"Choosing {chosen_deployment.name} (the first one) as "
                f"the default model deployment for model {model_name}"
            )
            hlog("If you want to use a different model deployment, please specify it explicitly.")
        return chosen_deployment.name

    # Some models are added but have no deployments yet.
    # In this case, we return None.
    return None


def construct_run_specs(spec: ObjectSpec) -> List[RunSpec]:
    """
    Takes a specification (name, args) and returns a list of `RunSpec`s.
    """
    # Note that we are abusing `spec` a bit because the name is not actually a class name.
    name = spec.class_name
    args = spec.args

    run_spec_function = get_run_spec_function(name)
    if run_spec_function is None:
        raise ValueError(f"Unknown run spec name: {name}")

    # Peel off the run expanders (e.g., model)
    expanders = [RUN_EXPANDERS[key](value) for key, value in args.items() if key in RUN_EXPANDERS]  # type: ignore
    args = dict((key, value) for key, value in args.items() if key not in RUN_EXPANDERS)

    run_specs: List[RunSpec] = [run_spec_function(**args)]

    # Apply expanders
    for expander in expanders:
        run_specs = [
            child_run_spec for parent_run_spec in run_specs for child_run_spec in expander.expand(parent_run_spec)
        ]

    def alter_run_spec(run_spec: RunSpec) -> RunSpec:
        if not run_spec.adapter_spec.model and not run_spec.adapter_spec.model_deployment:
            raise ValueError("At least one of model_deployment and model must be specified")
        elif not run_spec.adapter_spec.model and run_spec.adapter_spec.model_deployment:
            # Infer model from model deployment
            default_model_name = get_model_deployment(run_spec.adapter_spec.model_deployment).model_name
            if not default_model_name:
                default_model_name = run_spec.adapter_spec.model_deployment
            run_spec = dataclasses.replace(
                run_spec,
                adapter_spec=dataclasses.replace(run_spec.adapter_spec, model=default_model_name),
            )
        elif run_spec.adapter_spec.model and not run_spec.adapter_spec.model_deployment:
            # Infer model deployment from model
            default_model_deployment = get_default_model_deployment_for_model(run_spec.adapter_spec.model)
            if not default_model_deployment:
                raise ValueError(
                    f"Unknown model or no default model deployment found for model {run_spec.adapter_spec.model}"
                )
            run_spec = dataclasses.replace(
                run_spec,
                adapter_spec=dataclasses.replace(run_spec.adapter_spec, model_deployment=default_model_deployment),
            )

        # Both model and model_deployment should now be filled
        assert run_spec.adapter_spec.model_deployment
        assert run_spec.adapter_spec.model

        model: ModelMetadata = get_model_metadata(run_spec.adapter_spec.model)
        deployment: ModelDeployment = get_model_deployment(run_spec.adapter_spec.model_deployment)
        if run_spec.adapter_spec.model != deployment.model_name:
            raise ValueError(
                f"Invalid RunSpec: selected model deployment '{run_spec.adapter_spec.model_deployment}'"
                f"for model '{run_spec.adapter_spec.model}' but the model deployment is "
                f"for a different model '{deployment.model_name}'"
            )
        # For models that strip newlines, when we're generating, we need to set
        # the delimiter to be '###' so we stop properly.
        if NO_NEWLINES_TAG in model.tags and run_spec.adapter_spec.method in (
            ADAPT_GENERATION,
            ADAPT_MULTIPLE_CHOICE_JOINT,
        ):
            stop_expander = StopRunExpander(value="hash")
            run_spec = singleton(stop_expander.expand(run_spec))

        if NLG_PREFIX_TAG in model.tags:
            global_prefix_expander = GlobalPrefixRunExpander(value="nlg")
            run_spec = singleton(global_prefix_expander.expand(run_spec))

        if CHATML_MODEL_TAG in model.tags:
            chatml_expander = ChatMLRunExpander()
            run_spec = singleton(chatml_expander.expand(run_spec))

        # Anthropic prompts
        if ANTHROPIC_CLAUDE_1_MODEL_TAG in model.tags or ANTHROPIC_CLAUDE_2_MODEL_TAG in model.tags:
            run_spec = singleton(AnthropicRunExpander().expand(run_spec))

        # OpenAI prompts
        if OPENAI_CHATGPT_MODEL_TAG in model.tags:
            run_spec = singleton(OpenAIRunExpander().expand(run_spec))

        # Google prompts
        if GOOGLE_PALM_2_MODEL_TAG in model.tags or GOOGLE_GEMINI_MODEL_TAG in model.tags:
            run_spec = singleton(GoogleRunExpander().expand(run_spec))

        # IDEFICS instruct
        if IDEFICS_INSTRUCT_MODEL_TAG in model.tags:
            run_spec = singleton(IDEFICSInstructRunExpander().expand(run_spec))

        # Llava
        if LLAVA_MODEL_TAG in model.tags:
            run_spec = singleton(LlavaRunExpander().expand(run_spec))

        # For multiple choice
        if BUGGY_TEMP_0_TAG in model.tags and run_spec.adapter_spec.temperature == 0:
            increase_temperature_expander = IncreaseTemperatureRunExpander(value=1e-4)
            run_spec = singleton(increase_temperature_expander.expand(run_spec))

        return run_spec

    run_specs = [alter_run_spec(run_spec) for run_spec in run_specs]

    return run_specs
