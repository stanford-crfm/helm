import dataclasses
from typing import List

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.model_deployment_registry import (
    ModelDeployment,
    get_default_model_deployment_for_model,
    get_model_deployment,
)
from helm.benchmark.model_metadata_registry import (
    ANTHROPIC_CLAUDE_1_MODEL_TAG,
    ANTHROPIC_CLAUDE_2_MODEL_TAG,
    ANTHROPIC_CLAUDE_3_MODEL_TAG,
    BUGGY_TEMP_0_TAG,
    CHATML_MODEL_TAG,
    GOOGLE_GEMINI_PRO_VISION_V1_TAG,
    IDEFICS_INSTRUCT_MODEL_TAG,
    LLAVA_MODEL_TAG,
    OPEN_FLAMINGO_MODEL_TAG,
    NLG_PREFIX_TAG,
    NO_NEWLINES_TAG,
    VISION_LANGUAGE_MODEL_TAG,
    IDEFICS_MODEL_TAG,
    ModelMetadata,
    get_model_metadata,
)
from helm.benchmark.run_expander import (
    RUN_EXPANDERS,
    AnthropicClaude2RunExpander,
    AnthropicClaude3RunExpander,
    ChatMLRunExpander,
    GlobalPrefixRunExpander,
    IDEFICSInstructRunExpander,
    IncreaseTemperatureRunExpander,
    IncreaseMaxTokensRunExpander,
    LlavaRunExpander,
    ModelRunExpander,
    OpenFlamingoRunExpander,
    StopRunExpander,
)
from helm.benchmark.run_spec import RunSpec, get_run_spec_function
from helm.common.general import singleton
from helm.common.object_spec import ObjectSpec


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

    # If no model run expander was specified, add the model=all run expander
    if not any([expander for expander in expanders if isinstance(expander, ModelRunExpander)]):
        expanders.append(ModelRunExpander("all"))

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

        # Anthropic Claude 1 and 2 prompts
        if ANTHROPIC_CLAUDE_1_MODEL_TAG in model.tags or ANTHROPIC_CLAUDE_2_MODEL_TAG in model.tags:
            run_spec = singleton(AnthropicClaude2RunExpander().expand(run_spec))

        # Anthropic Claude 3
        if ANTHROPIC_CLAUDE_3_MODEL_TAG in model.tags:
            run_spec = singleton(AnthropicClaude3RunExpander().expand(run_spec))

        # Google Gemini Vision v1.0 returns an empty completion or throws an error if max_tokens is 1
        if (
            VISION_LANGUAGE_MODEL_TAG in model.tags
            and GOOGLE_GEMINI_PRO_VISION_V1_TAG in model.tags
            and run_spec.adapter_spec.max_tokens == 1
        ):
            run_spec = singleton(IncreaseMaxTokensRunExpander(value=1).expand(run_spec))

        if model.name == "openai/o1-2024-12-17":
            # From https://platform.openai.com/docs/guides/reasoning,
            # "OpenAI recommends reserving at least 25,000 tokens for reasoning and outputs when you start
            # experimenting with these models. As you become familiar with the number of reasoning tokens your
            # prompts require, you can adjust this buffer accordingly."
            run_spec = singleton(IncreaseMaxTokensRunExpander(value=25_000).expand(run_spec))

        # IDEFICS special handling
        if IDEFICS_MODEL_TAG in model.tags:
            if IDEFICS_INSTRUCT_MODEL_TAG in model.tags:
                run_spec = singleton(IDEFICSInstructRunExpander().expand(run_spec))

        # Llava
        if LLAVA_MODEL_TAG in model.tags:
            run_spec = singleton(LlavaRunExpander().expand(run_spec))

        # OpenFlamingo
        if OPEN_FLAMINGO_MODEL_TAG in model.tags:
            run_spec = singleton(OpenFlamingoRunExpander().expand(run_spec))

        # For multiple choice
        if BUGGY_TEMP_0_TAG in model.tags and run_spec.adapter_spec.temperature == 0:
            increase_temperature_expander = IncreaseTemperatureRunExpander(value=1e-4)
            run_spec = singleton(increase_temperature_expander.expand(run_spec))

        # MedLM-Large
        if run_spec.adapter_spec.model == "google/medlm-large":
            run_spec = singleton(StopRunExpander("none").expand(run_spec))

        return run_spec

    run_specs = [alter_run_spec(run_spec) for run_spec in run_specs]

    return run_specs
