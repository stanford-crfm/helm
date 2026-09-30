import os
import pytest

from helm.common.object_spec import parse_object_spec
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run_spec_factory import construct_run_specs


def list_fnames():
    base_path = os.path.dirname(__file__)
    return [os.path.join(base_path, fname) for fname in os.listdir(base_path) if fname.endswith(".conf")]


class TestRunEntry:
    """Read all the run entries and make sure they parse and we can instantiate them."""

    @pytest.mark.parametrize("fname", list_fnames())
    def test_parse_all_specs(self, fname: str):
        """Check that every run entry parses into an ObjectSpec.

        This stops short of constructing the run specs, so it performs no downloads and
        runs in a few seconds across all run entry files. It catches malformed .conf
        files and unparseable run entry descriptions.
        """
        run_entries = read_run_entries([fname])
        assert run_entries.entries, f"No run entries were read from {os.path.basename(fname)}"
        for entry in run_entries.entries:
            parse_object_spec(entry.description)

    @pytest.mark.scenarios
    @pytest.mark.parametrize("fname", list_fnames())
    def test_construct_all_specs(self, fname: str):
        """Check that every run entry can also be instantiated into a RunSpec.

        Constructing run specs downloads scenario data, so this is marked `scenarios`
        and is deselected by the default addopts. Run it with `pytest -m scenarios`.
        """
        run_entries = read_run_entries([fname])
        for entry in run_entries.entries:
            construct_run_specs(parse_object_spec(entry.description))
