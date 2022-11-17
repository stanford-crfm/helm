import os
from helm.common.object_spec import parse_object_spec
from helm.benchmark.presentation.run_entry import read_run_entries
from helm.benchmark.run_specs import construct_run_specs


def test_read_all_specs():
    """Read all the run entries and make sure they parse and we can instantiate them."""
    base_path = os.path.dirname(__file__)
    for fname in os.listdir(base_path):
        if fname.endswith(".conf"):
            run_entries = read_run_entries([os.path.join(base_path, fname)])
            for entry in run_entries.entries:
                construct_run_specs(parse_object_spec(entry.description))
