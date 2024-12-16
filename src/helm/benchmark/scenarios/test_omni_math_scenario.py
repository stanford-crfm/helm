import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.omni_math_scenario import OmniMATHScenario
from helm.benchmark.scenarios.scenario import Input, TEST_SPLIT


@pytest.mark.scenarios
def test_omni_math_scenario_get_instances():
    omni_math_scenario = OmniMATHScenario()
    with TemporaryDirectory() as tmpdir:
        instances = omni_math_scenario.get_instances(tmpdir)
    assert len(instances) == 4428
    assert instances[0].input == Input(
        text=(
            "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $, "
            "so that there exists $x_{ij}(1\\le i ,j\\le n)$ satisfying:\n(1)For every "
            "$1\\le i ,j\\le n, x_{ij}=max\\{x_{i1},x_{i2},...,x_{ij}\\} $ or $ x_{ij}="
            "max\\{x_{1j},x_{2j},...,x_{ij}\\}.$\n(2)For every $1\\le i \\le n$, there "
            "are at most $m$ indices $k$ with $x_{ik}=max\\{x_{i1},x_{i2},...,x_{ik}\\}."
            "$\n(3)For every $1\\le j \\le n$, there are at most $m$ indices $k$ with "
            "$x_{kj}=max\\{x_{1j},x_{2j},...,x_{kj}\\}.$"
        )
    )
    assert instances[0].split == TEST_SPLIT
    assert instances[0].references
    assert instances[0].references[0].output.text == "1 + \\left\\lceil \\frac{n}{2} \\right\\rceil"
