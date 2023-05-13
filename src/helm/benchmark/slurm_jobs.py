import os
import re
import subprocess
from typing import Dict, Set, Union

from simple_slurm import Slurm


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

# TODO: Make default Slurm arguments configurable.
# TODO: Request more memory for MSMARCO runs and request GPUs for CUDA runs.
_DEFAULT_SLURM_ARGS: Dict[str, Union[str, int]] = {
    "account": "nlp",
    "cpus_per_task": 2,
    "mem": "8G",
    "gres": "gpu:0",
    "open_mode": "append",
    "partition": "john",
    "time": "14-0",
    "mail_type": "END",
}


def submit_slurm_job(command: str, job_name: str, output_path: str) -> int:
    """Submit a Slurm job."""
    slurm = Slurm(**_DEFAULT_SLURM_ARGS)
    slurm.add_arguments(job_name=job_name, output=output_path, chdir=os.getcwd())
    # Uncomment this to get notification emails from Slurm for Slurm worker jobs.
    # slurm.set_mail_user(os.getenv("USER"))
    return slurm.sbatch(command)


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


def cancel_slurm_job(job_id: int) -> None:
    """Cancel a Slurm job."""
    try:
        # Note: scancel will execute successfully on any existing job, even if the job is already terminated.
        subprocess.check_output(f"scancel {job_id}", stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        # Default CalledProcessError message doesn't have output, so re-raise here to include the output.
        raise Exception(f"{str(e)} output: {e.output}")
