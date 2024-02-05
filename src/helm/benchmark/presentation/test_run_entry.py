import os
import pytest

from helm.common.object_spec import parse_object_spec
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run_specs import construct_run_specs
from helm.benchmark import heim_run_specs  # noqa
from helm.benchmark import vlm_run_specs  # noqa


def list_fnames():
    base_path = os.path.dirname(__file__)
    # Temporarily disable test_read_all_specs() for CLEVA because CLEVA scenarios are broken due to a missing data file.
    # TODO(#2303): Re-enable test_read_all_specs() for CLEVA.
    return [
        os.path.join(base_path, fname)
        for fname in os.listdir(base_path)
        if fname.endswith(".conf") and "cleva" not in fname
    ]


class TestRunEntry:
    """Read all the run entries and make sure they parse and we can instantiate them."""

    @pytest.mark.parametrize("fname", list_fnames())
    def test_read_all_specs(self, fname: str):
        run_entries = read_run_entries([fname])
        for entry in run_entries.entries:
            construct_run_specs(parse_object_spec(entry.description))
