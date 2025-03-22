import argparse
from dataclasses import replace
import re
from typing import List, Optional

from helm.benchmark import model_metadata_registry
from helm.benchmark.presentation.run_entry import RunEntry, read_run_entries
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack
from helm.common.authentication import Authentication
from helm.common.object_spec import parse_object_spec
from helm.proxy.services.remote_service import create_authentication, add_service_args

from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.runner import RunSpec, set_benchmark_output_path
from helm.benchmark.run_spec_factory import construct_run_specs
from helm.common.reeval_parameters import ReevalParameters
from helm.benchmark.run import (
    run_benchmarking,
    validate_args,
)


def run_entries_to_run_specs(
    run_entries: List[RunEntry],
    max_eval_instances: Optional[int] = None,
    num_train_trials: Optional[int] = None,
    models_to_run: Optional[List[str]] = None,
    groups_to_run: Optional[List[str]] = None,
    priority: Optional[int] = None,
    model_ability: Optional[float] = None,
    max_samples: Optional[int] = None,
    metric_name: Optional[str] = None,
) -> List[RunSpec]:
    """Runs RunSpecs given a list of RunSpec descriptions."""
    run_specs: List[RunSpec] = []
    for entry in run_entries:
        # Filter by priority
        if priority is not None and entry.priority > priority:
            continue

        for run_spec in construct_run_specs(parse_object_spec(entry.description)):
            # Filter by models
            if models_to_run and run_spec.adapter_spec.model not in models_to_run:
                continue

            # Filter by groups
            if groups_to_run and not any(group in groups_to_run for group in run_spec.groups):
                continue

            # Modify AdapterSpec
            adapter_spec: AdapterSpec = run_spec.adapter_spec
            if max_eval_instances is not None and adapter_spec.max_eval_instances is None:
                adapter_spec = replace(adapter_spec, max_eval_instances=max_eval_instances)

            if adapter_spec.max_train_instances == 0:
                adapter_spec = replace(adapter_spec, num_train_trials=1)
            elif num_train_trials is not None:
                adapter_spec = replace(adapter_spec, num_train_trials=num_train_trials)

            # Add reeval_parameters
            adapter_spec = replace(
                adapter_spec,
                reeval_parameters=ReevalParameters(
                    model_ability=model_ability,
                    max_samples=max_samples,
                    metric_name=metric_name,
                ),
            )

            run_spec = replace(run_spec, adapter_spec=adapter_spec)

            # Append groups
            if entry.groups is not None:
                groups_name: str = "" if len(entry.groups) == 0 else f",groups={'-'.join(sorted(entry.groups))}"
                run_spec = replace(run_spec, name=run_spec.name + groups_name, groups=run_spec.groups + entry.groups)

            run_specs.append(run_spec)

    return run_specs


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where to save all the output", default="benchmark_output"
    )
    parser.add_argument("-n", "--num-threads", type=int, help="Max number of threads to make requests", default=4)
    parser.add_argument(
        "--skip-instances",
        action="store_true",
        help="Skip creation of instances (basically do nothing but just parse everything).",
    )
    parser.add_argument(
        "--cache-instances",
        action="store_true",
        help="Save generated instances input to model to disk. If already cached, read instances from file.",
    )
    parser.add_argument(
        "--cache-instances-only",
        action="store_true",
        help="Generate and save instances for scenario ONLY (i.e. do not evaluate models on instances).",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Skip execution, only output scenario states and estimate token usage.",
    )
    parser.add_argument(
        "-t",
        "--num-train-trials",
        type=int,
        help="Number of trials where each trial samples a different set of in-context examples. "
        "Overrides the value in Adapter spec.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this run belongs to (default is today's date).",
        required=True,
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="If running locally, the path for `ServerService`.",
        default="prod_env",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help="If non-empty, the URL of the MongoDB database that will be used for caching instead of SQLite",
        default="",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="If true, the request-response cache for model clients and tokenizers will be disabled.",
    )


@htrack(None)
def main():
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument(
        "-c",
        "--conf-paths",
        nargs="+",
        help="Where to read RunSpecs to run from",
        default=[],
    )
    parser.add_argument(
        "--models-to-run",
        nargs="+",
        help="Only RunSpecs with these models specified. If no model is specified, runs with all models.",
        default=None,
    )
    parser.add_argument(
        "--groups-to-run",
        nargs="+",
        help="Only RunSpecs with these (scenario) groups specified. " "If no group is specified, runs with all groups.",
        default=None,
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="Fail and exit immediately if a particular RunSpec fails.",
    )
    parser.add_argument(
        "--skip-completed-runs",
        action="store_true",
        help="Skip RunSpecs that have completed i.e. output files exists.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Run RunSpecs with priority less than or equal to this number. "
        "If a value for --priority is not specified, run on everything",
    )
    parser.add_argument(
        "--run-specs",
        nargs="*",
        help="DEPRECATED: Use --run-entries instead. Will be removed in a future release. "
        "Specifies run entries to run.",
        default=[],
    )
    parser.add_argument("-r", "--run-entries", nargs="*", help="Specifies run entries to run", default=[])
    parser.add_argument(
        "--enable-huggingface-models",
        nargs="+",
        default=[],
        help="Experimental: Enable using AutoModelForCausalLM models from Hugging Face Model Hub. "
        "Format: namespace/model_name[@revision]",
    )
    parser.add_argument(
        "--enable-local-huggingface-models",
        nargs="+",
        default=[],
        help="Experimental: Enable using AutoModelForCausalLM models from a local path.",
    )
    parser.add_argument(
        "--runner-class-name",
        type=str,
        default=None,
        help="Full class name of the Runner class to use. If unset, uses the default Runner.",
    )
    # reeval parameters
    parser.add_argument(
        "--model-ability",
        type=float,
        default=0.0,
        help="The initial ability of the model for reeval evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate in reeval mode.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="The main metric name for the scenario.",
    )
    add_run_args(parser)
    args = parser.parse_args()
    validate_args(args)
    register_builtin_configs_from_helm_package()
    register_configs_from_directory(args.local_path)

    if args.enable_huggingface_models:
        from helm.benchmark.huggingface_registration import register_huggingface_hub_model_from_flag_value

        for huggingface_model_name in args.enable_huggingface_models:
            register_huggingface_hub_model_from_flag_value(huggingface_model_name)

    if args.enable_local_huggingface_models:
        from helm.benchmark.huggingface_registration import register_huggingface_local_model_from_flag_value

        for huggingface_model_path in args.enable_local_huggingface_models:
            register_huggingface_local_model_from_flag_value(huggingface_model_path)

    run_entries: List[RunEntry] = []
    if args.conf_paths:
        run_entries.extend(read_run_entries(args.conf_paths).entries)
    if args.run_entries:
        run_entries.extend(
            [RunEntry(description=description, priority=1, groups=None) for description in args.run_entries]
        )
    # TODO: Remove this eventually.
    if args.run_specs:
        run_entries.extend(
            [RunEntry(description=description, priority=1, groups=None) for description in args.run_specs]
        )

    # Must set benchmark output path before getting RunSpecs,
    # because run spec functions can use the benchmark output directory for caching.
    ensure_directory_exists(args.output_path)
    set_benchmark_output_path(args.output_path)

    # Validate the --models-to-run flag
    if args.models_to_run:
        all_models = set(model_metadata_registry.get_all_models())
        for model_to_run in args.models_to_run:
            if model_to_run not in all_models:
                raise Exception(f"Unknown model '{model_to_run}' passed to --models-to-run")
    else:
        model_expander_pattern = re.compile(
            r"\bmodel=(?:all|text_code|text|code|instruction_following|full_functionality_text|limited_functionality_text)\b"  # noqa: E501
        )
        if any(model_expander_pattern.search(run_entry.description) for run_entry in run_entries):
            raise Exception("--models-to-run must be set if the `models=` run expander expands to multiple models")

    run_specs = run_entries_to_run_specs(
        run_entries=run_entries,
        num_train_trials=args.num_train_trials,
        models_to_run=args.models_to_run,
        groups_to_run=args.groups_to_run,
        priority=args.priority,
        model_ability=args.model_ability,
        max_samples=args.max_samples,
        metric_name=args.metric_name,
    )
    hlog(f"{len(run_entries)} entries produced {len(run_specs)} run specs")

    if len(run_specs) == 0:
        hlog("There were no RunSpecs or they got filtered out.")
        return

    auth: Authentication = (
        Authentication("") if args.skip_instances or not args.server_url else create_authentication(args)
    )

    run_benchmarking(
        run_specs=run_specs,
        auth=auth,
        url=args.server_url,
        local_path=args.local_path,
        num_threads=args.num_threads,
        output_path=args.output_path,
        suite=args.suite,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        cache_instances=args.cache_instances,
        cache_instances_only=args.cache_instances_only,
        skip_completed_runs=args.skip_completed_runs,
        exit_on_error=args.exit_on_error,
        runner_class_name=args.runner_class_name,
        mongo_uri=args.mongo_uri,
        disable_cache=args.disable_cache,
    )

    if args.run_specs:
        hlog(
            "WARNING: The --run-specs flag is deprecated and will be removed in a future release. "
            "Use --run-entries instead."
        )

    hlog("Done.")


if __name__ == "__main__":
    main()
