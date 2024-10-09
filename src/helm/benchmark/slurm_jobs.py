import re
import subprocess
from typing import Mapping, Set, Union

from retrying import retry

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from simple_slurm import Slurm
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["slurm"])


class SlurmJobState:
    # TODO: Convert to StrEnum after upgrading to Python 3.11
    # Non-exhaustive list of Slurm job states.
    # See: https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES

    # Healthy non-terminal states
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    COMPLETING = "COMPLETING"

    # Successful terminal state
    COMPLETED = "COMPLETED"

    # Failure terminal states
    BOOT_FAIL = "BOOT_FAIL"
    CANCELLED = "CANCELLED"
    DEADLINE = "DEADLINE"
    FAILED = "FAILED"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    PREEMPTED = "PREEMPTED"


ACTIVE_SLURM_JOB_STATES: Set[str] = set(
    [
        SlurmJobState.PENDING,
        SlurmJobState.RUNNING,
        SlurmJobState.SUSPENDED,
    ]
)
"""Slurm job active (i.e. healthy non-terminal) states."""


FAILURE_SLURM_JOB_STATES: Set[str] = set(
    [
        SlurmJobState.BOOT_FAIL,
        SlurmJobState.CANCELLED,
        SlurmJobState.DEADLINE,
        SlurmJobState.FAILED,
        SlurmJobState.NODE_FAIL,
        SlurmJobState.OUT_OF_MEMORY,
        SlurmJobState.PREEMPTED,
    ]
)
"""Slurm job failure terminal states."""

TERMINAL_SLURM_JOB_STATES: Set[str] = set([SlurmJobState.COMPLETED]) | FAILURE_SLURM_JOB_STATES
"""Slurm job terminal states."""


def submit_slurm_job(command: str, slurm_args: Mapping[str, Union[str, int]]) -> int:
    """Submit a Slurm job."""
    slurm = Slurm(**slurm_args)
    return slurm.sbatch(command)


@retry(
    wait_incrementing_start=5 * 1000,  # 5 seconds
    wait_incrementing_increment=5 * 1000,  # 5 seconds
    stop_max_attempt_number=5,
)
def get_slurm_job_state(job_id: int) -> str:
    """Get the state of a Slurm job."""
    try:
        scontrol_output = subprocess.check_output(f"scontrol show job {job_id}", stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        # Default CalledProcessError message doesn't have output, so re-raise here to include the output.
        raise Exception(f"{str(e)} output: {e.output}")
    search_result = re.search("JobState=(\w+)", scontrol_output.decode())
    if not search_result:
        raise Exception(f"Could not extract JobState from scontrol: {scontrol_output.decode()}")
    return search_result.group(1)


@retry(
    wait_incrementing_start=5 * 1000,  # 5 seconds
    wait_incrementing_increment=5 * 1000,  # 5 seconds
    stop_max_attempt_number=5,
)
def cancel_slurm_job(job_id: int) -> None:
    """Cancel a Slurm job."""
    try:
        # Note: scancel will execute successfully on any existing job, even if the job is already terminated.
        subprocess.check_output(f"scancel {job_id}", stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        # Default CalledProcessError message doesn't have output, so re-raise here to include the output.
        raise Exception(f"{str(e)} output: {e.output}")
