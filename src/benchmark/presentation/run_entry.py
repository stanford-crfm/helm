from dataclasses import dataclass
from typing import Optional, List
import dacite

from common.general import parse_hocon
from common.hierarchical_logger import hlog


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


def read_run_entries(path: str) -> RunEntries:
    """Read a HOCON file `path` and return the `RunEntry`s."""
    with open(path) as f:
        raw = parse_hocon(f.read())
    run_entries = dacite.from_dict(RunEntries, raw)
    hlog(f"Read {len(run_entries.entries)} run entries from {path}")
    return run_entries
