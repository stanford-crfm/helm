import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.medcalc_bench_scenario import (
    MedCalcBenchScenario,
    MedCalcBenchV1_0Scenario,
    MedCalcBenchV1_1Scenario,
    MedCalcBenchV1_2Scenario,
)


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
        scenario = MedCalcBenchV1_0Scenario()
        instances = scenario.get_instances(tmpdir)

        assert instances[0].split == "test"


@pytest.mark.scenarios
def test_medcalc_bench_v1_1_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the MedCalc-Bench scenario
        scenario = MedCalcBenchV1_1Scenario()
        instances = scenario.get_instances(tmpdir)

        assert instances[0].split == "test"


@pytest.mark.scenarios
def test_medcalc_bench_v1_2_scenario():
    with TemporaryDirectory() as tmpdir:
        # Test for the MedCalc-Bench scenario
        scenario = MedCalcBenchV1_2Scenario()
        instances = scenario.get_instances(tmpdir)

        assert instances[0].split == "test"
