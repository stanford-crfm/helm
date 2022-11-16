from dataclasses import dataclass
from typing import Optional, List
import dacite

from helm.common.general import parse_hocon
from helm.common.hierarchical_logger import hlog


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
