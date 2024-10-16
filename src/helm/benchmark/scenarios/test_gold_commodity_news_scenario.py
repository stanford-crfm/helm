import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.gold_commodity_news_scenario import GoldCommodityNewsScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_legalbench_scenario():
    scenario = GoldCommodityNewsScenario(category="price_or_not")
    with TemporaryDirectory() as tmpdir:
        instances = scenario.get_instances(tmpdir)
    assert len(instances) == 11412
    assert instances[0].input == Input(text="april gold down 20 cents to settle at $1,116.10/oz")
    assert instances[0].references == [
        Reference(output=Output(text="Yes"), tags=[CORRECT_TAG]),
    ]
    assert instances[0].split == "test"
