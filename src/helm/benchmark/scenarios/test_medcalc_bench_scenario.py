import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.medcalc_bench_scenario import MedCalcBenchScenario


@pytest.mark.scenarios
def test_medcalc_bench_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the MedCalc-Bench scenario
        scenario = MedCalcBenchScenario()
        instances = scenario.get_instances(tmpdir)

        assert instances[0].split == "test"


@pytest.mark.scenarios
def test_medcalc_bench_v1_0_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the MedCalc-Bench scenario
        scenario = MedCalcBenchScenario()
        instances = scenario.get_instances(tmpdir, "v1.0")

        assert instances[0].split == "test"


@pytest.mark.scenarios
def test_medcalc_bench_v1_1_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the MedCalc-Bench scenario
        scenario = MedCalcBenchScenario()
        instances = scenario.get_instances(tmpdir, "v1.1")

        assert instances[0].split == "test"


@pytest.mark.scenarios
def test_medcalc_bench_v1_2_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the MedCalc-Bench scenario
        scenario = MedCalcBenchScenario()
        instances = scenario.get_instances(tmpdir, "v1.2")

        assert instances[0].split == "test"
