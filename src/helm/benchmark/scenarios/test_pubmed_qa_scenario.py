from helm.benchmark.scenarios.pubmed_qa_scenario import PubMedQAScenario


def test_pubmed_qa_main_metric_uses_quasi_exact_match():
    scenario = PubMedQAScenario()

    assert scenario.get_metadata().main_metric == "quasi_exact_match"
