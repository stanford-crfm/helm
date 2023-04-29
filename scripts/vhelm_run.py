from dataclasses import dataclass
from typing import List, Optional, Tuple
import argparse
import dacite
import os

from helm.common.general import parse_hocon, ensure_directory_exists
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.proxy.models import get_model, get_models_with_tag, TEXT_TO_IMAGE_MODEL_TAG, Model


NLP_RUN: str = "nlprun"
DEFAULT_NLP_RUN: str = "-a vhelm -c 4 --memory 64g -w /nlp/scr4/nlp/crfm/benchmarking/benchmarking"
DEFAULT_NLP_RUN_CPU_ARGS: str = f"{DEFAULT_NLP_RUN} -g 0 --exclude john17"
# jag 27, 29, 34 started the run, but did nothing. Saw CUDA OOM with 28.
DEFAULT_NLP_RUN_GPU_ARGS: str = f"{DEFAULT_NLP_RUN} -g 1 --exclude jagupard[10-25,27,28,29,34],sphinx3"
MONGODB_MACHINE: str = "john13"
DEFAULT_HELM_ARGS: str = (
    f"--num-train-trials 1 --local -n 1 --mongo-uri='mongodb://crfm-benchmarking:kindling-vespers"
    f"@{MONGODB_MACHINE}/crfm-models'"
)

CUDA_OOM_ERROR: str = "CUDA out of memory"


# Copied from run_entry.py. Using the method directly results in an error.
@dataclass(frozen=True)
class RunEntry:
    """Represents something that we want to run."""

    # Gets expanded into a list of `RunSpec`s.
    description: str

    # Priority for this run spec (1 is highest priority, 5 is lowest priority)
    priority: int

    # Additional groups to add to the run spec
    groups: Optional[List[str]]


@dataclass(frozen=True)
class RunEntries:
    entries: List[RunEntry]


def merge_run_entries(run_entries1: RunEntries, run_entries2: RunEntries):
    return RunEntries(run_entries1.entries + run_entries2.entries)


def read_run_entries(paths: List[str]) -> RunEntries:
    """Read a HOCON file `path` and return the `RunEntry`s."""
    run_entries = RunEntries([])
    for path in paths:
        with open(path) as f:
            raw = parse_hocon(f.read())
        run_entries = merge_run_entries(run_entries, dacite.from_dict(RunEntries, raw))
        hlog(f"Read {len(run_entries.entries)} run entries from {path}")
    return run_entries


@htrack(None)
def queue_jobs(
    conf_path: str,
    suite: str,
    models_to_run: List[str],
    priority: int = 2,
    dry_run: bool = False,
    use_sphinx: bool = False,
    machine: Optional[str] = None,
):
    # Create a run directory at benchmark_output/runs/<suite>
    suite_path: str = os.path.join("benchmark_output", "runs", suite)
    ensure_directory_exists(suite_path)
    logs_path: str = os.path.join(suite_path, "logs")
    ensure_directory_exists(logs_path)
    confs_path: str = os.path.join(suite_path, "confs")
    ensure_directory_exists(confs_path)

    # Read the RunSpecs and split each RunSpec into its own file
    confs: List[Tuple[str, str]] = []
    run_entries = read_run_entries([conf_path])
    for entry in run_entries.entries:
        if entry.priority > priority:
            continue

        description: str = entry.description
        conf_path = os.path.join(confs_path, f"{description}.conf")
        with open(conf_path, "w") as f:
            f.write(f'entries: [{{description: "{description}", priority: {priority}}}]')
        confs.append((description, conf_path))

    # Run all models on all RunSpecs if `models_to_run` is empty
    models: List[Model] = (
        get_models_with_tag(TEXT_TO_IMAGE_MODEL_TAG)
        if len(models_to_run) == 0
        else [get_model(model_name) for model_name in models_to_run]
    )
    for model in models:
        for i, (description, conf_path) in enumerate(confs):
            # Construct command here
            job_name: str = f"{model.engine}_{description}"
            log_path: str = os.path.join(logs_path, f"{job_name}.log")

            nlp_run_gpu_args: str = DEFAULT_NLP_RUN_GPU_ARGS
            if machine is not None:
                nlp_run_gpu_args += f" -m {machine}"

                if "sphinx" in machine:
                    nlp_run_gpu_args += " -p high "
            elif use_sphinx:
                nlp_run_gpu_args += " -q sphinx "

            swiss_army_port: str = ""
            if model.name == "thudm/cogview2":
                swiss_army_port = f"MASTER_PORT={60_000 + i} "

            command: str = (
                f"{NLP_RUN} {nlp_run_gpu_args} --job-name {job_name} '{swiss_army_port}helm-run {DEFAULT_HELM_ARGS} "
                f"--suite {suite} --conf-paths {conf_path} --models-to-run {model.name} --priority {priority} "
                f"> {log_path} 2>&1'"
            )
            hlog(command)
            if not dry_run:
                os.system(command)

    # Inform what to run next: summarize + copy
    hlog("\n\nTo check on the runs:\n")
    hlog(
        f"scp -r tonyhlee@scdt.stanford.edu:/nlp/scr4/nlp/crfm/benchmarking/benchmarking/"
        f"benchmark_output/runs/{suite}/logs ."
    )
    hlog("\npython3 scripts/vhelm_run.py --check")
    hlog("\n\nRun the following command once all the runs complete:\n")
    command = (
        f"helm-summarize --suite {suite} > {os.path.join(logs_path, 'summarize.log')} 2>&1 && "
        f"sh scripts/create-www-vhelm.sh {suite} > {os.path.join(logs_path, 'upload.log')} 2>&1"
    )
    hlog(f"{NLP_RUN} {DEFAULT_NLP_RUN_CPU_ARGS} --job-name {suite}-summarize-upload '{command}'\n")


@htrack(None)
def check(suite: str):
    suite_path: str = os.path.join("benchmark_output", "runs", suite)
    logs_path: str = os.path.join(suite_path, "logs")
    cuda_oom_logs: List[str] = []

    for log_file in os.listdir(logs_path):
        if not log_file.endswith("log"):
            continue

        log_path: str = os.path.join(logs_path, log_file)
        with open(log_path, "r") as f:
            try:
                log_content: str = f.read()
                if CUDA_OOM_ERROR in log_content:
                    cuda_oom_logs.append(log_path)
                elif "Done.\n" not in log_content:
                    hlog(f"tail -f {log_path}")
            except UnicodeDecodeError as e:
                hlog(f"Error reading {log_path}: {e}")

    if len(cuda_oom_logs) > 0:
        with htrack_block(f"\nLog files with {CUDA_OOM_ERROR}:"):
            for i, log_path in enumerate(cuda_oom_logs):
                hlog(f"{i+1}. {log_path}")
    hlog("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf-path",
        help="Where to read RunSpecs to run from",
        default="src/helm/benchmark/presentation/run_specs_vhelm.conf",
    )
    parser.add_argument(
        "--models-to-run",
        nargs="+",
        help="Only RunSpecs with these models specified. If no model is specified, runs with all models.",
        default=[],
    )
    parser.add_argument("--suite", help="Name of the suite", default="vhelm")
    parser.add_argument("--priority", default=2)
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips execution.",
    )
    parser.add_argument(
        "--use-sphinx",
        action="store_true",
        default=None,
        help="Uses Sphinx machines if set.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=None,
        help="Checks logs",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="If specified, runs on that specific machine",
    )
    args = parser.parse_args()

    if args.check:
        check(args.suite)
    else:
        queue_jobs(
            conf_path=args.conf_path,
            suite=args.suite,
            models_to_run=args.models_to_run,
            priority=args.priority,
            dry_run=args.dry_run,
            use_sphinx=args.use_sphinx,
            machine=args.machine,
        )
