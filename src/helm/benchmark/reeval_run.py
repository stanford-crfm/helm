import argparse
from dataclasses import replace
import re
from typing import List

from helm.benchmark import model_metadata_registry
from helm.benchmark.presentation.run_entry import RunEntry, read_run_entries
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack
from helm.common.authentication import Authentication
from helm.proxy.services.remote_service import create_authentication, add_service_args

from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.runner import set_benchmark_output_path
from helm.common.reeval_parameters import REEvalParameters
from helm.benchmark.run import (
    run_benchmarking,
    validate_args,
    add_run_args,
    run_entries_to_run_specs,
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
    # reeval parameters
    parser.add_argument(
        "--model-ability",
        type=float,
        default=0.0,
        help="The initial ability of the model for reeval evaluation.",
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
        max_eval_instances=args.max_eval_instances,
        num_train_trials=args.num_train_trials,
        models_to_run=args.models_to_run,
        groups_to_run=args.groups_to_run,
        priority=args.priority,
    )
    hlog(f"{len(run_entries)} entries produced {len(run_specs)} run specs")

    if len(run_specs) == 0:
        hlog("There were no RunSpecs or they got filtered out.")
        return

    # Add reeval_parameters
    run_specs = [
        replace(
            run_spec,
            adapter_spec=replace(
                run_spec.adapter_spec, reeval_parameters=REEvalParameters(model_ability=args.model_ability)
            ),
        )
        for run_spec in run_specs
    ]

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
        runner_class_name="helm.benchmark.reeval_runner.REEvalRunner",
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
