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
    def test_read_all_specs(self, fname: str):
        pytest.skip("Skipping slow tests")
        run_entries = read_run_entries([fname])
        for entry in run_entries.entries:
            construct_run_specs(parse_object_spec(entry.description))
