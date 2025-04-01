import argparse
from dataclasses import replace
import os
import re
from typing import List, Optional


from helm.benchmark import model_metadata_registry
from helm.benchmark.presentation.run_entry import RunEntry, read_run_entries
from helm.common.cache_backend_config import MongoCacheBackendConfig, SqliteCacheBackendConfig
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.common.authentication import Authentication
from helm.common.object_spec import parse_object_spec, get_class_by_name
from helm.proxy.services.remote_service import create_authentication, add_service_args
from helm.proxy.services.service import CACHE_DIR

from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.runner import Runner, RunSpec, LATEST_SYMLINK, set_benchmark_output_path
from helm.benchmark.run_spec_factory import construct_run_specs


def run_entries_to_run_specs(
    run_entries: List[RunEntry],
    max_eval_instances: Optional[int] = None,
    num_train_trials: Optional[int] = None,
    models_to_run: Optional[List[str]] = None,
    groups_to_run: Optional[List[str]] = None,
    priority: Optional[int] = None,
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

            run_spec = replace(run_spec, adapter_spec=adapter_spec)

            # Append groups
            if entry.groups is not None:
                groups_name: str = "" if len(entry.groups) == 0 else f",groups={'-'.join(sorted(entry.groups))}"
                run_spec = replace(run_spec, name=run_spec.name + groups_name, groups=run_spec.groups + entry.groups)

            run_specs.append(run_spec)

    return run_specs


def run_benchmarking(
    run_specs: List[RunSpec],
    auth: Authentication,
    url: Optional[str],
    local_path: str,
    num_threads: int,
    output_path: str,
    suite: str,
    dry_run: bool,
    skip_instances: bool,
    cache_instances: bool,
    cache_instances_only: bool,
    skip_completed_runs: bool,
    exit_on_error: bool,
    runner_class_name: Optional[str],
    mongo_uri: Optional[str] = None,
    disable_cache: Optional[bool] = None,
) -> List[RunSpec]:
    """Runs RunSpecs given a list of RunSpec descriptions."""
    sqlite_cache_backend_config: Optional[SqliteCacheBackendConfig] = None
    mongo_cache_backend_config: Optional[MongoCacheBackendConfig] = None

    if not disable_cache:
        if mongo_uri:
            mongo_cache_backend_config = MongoCacheBackendConfig(mongo_uri)
        else:
            sqlite_cache_path = os.path.join(local_path, CACHE_DIR)
            ensure_directory_exists(sqlite_cache_path)
            sqlite_cache_backend_config = SqliteCacheBackendConfig(sqlite_cache_path)

    execution_spec = ExecutionSpec(
        auth=auth,
        url=url,
        local_path=local_path,
        parallelism=num_threads,
        dry_run=dry_run,
        sqlite_cache_backend_config=sqlite_cache_backend_config,
        mongo_cache_backend_config=mongo_cache_backend_config,
    )
    with htrack_block("run_specs"):
        for run_spec in run_specs:
            hlog(run_spec)
    runner_cls = get_class_by_name(runner_class_name) if runner_class_name else Runner
    runner: Runner = runner_cls(
        execution_spec,
        output_path,
        suite,
        skip_instances,
        cache_instances,
        cache_instances_only,
        skip_completed_runs,
        exit_on_error,
    )
    runner.run_all(run_specs)
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
        "-m",
        "--max-eval-instances",
        type=int,
        required=True,
        help="Maximum number of instances to evaluate on, overrides the value in Adapter spec.",
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


def validate_args(args):
    assert args.suite != LATEST_SYMLINK, f"Suite name can't be '{LATEST_SYMLINK}'"
    if args.cache_instances_only:
        assert args.cache_instances, "If --cache-instances-only is set, --cache-instances must also be set."


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
        model_expander_wildcard_pattern = re.compile(
            r"\bmodel=(?:all|text_code|text|code|instruction_following|full_functionality_text|limited_functionality_text)\b"  # noqa: E501
        )
        if any(model_expander_wildcard_pattern.search(run_entry.description) for run_entry in run_entries):
            raise Exception("--models-to-run must be set if the `models=` run expander expands to multiple models")

        model_expander_pattern = re.compile(r"\bmodel=\b")
        if not any(model_expander_pattern.search(run_entry.description) for run_entry in run_entries):
            raise Exception("--models-to-run must be set if the `models=` run expander is omitted")

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
