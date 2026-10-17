"""Offline test for the zBench IVF scenario and grader: one real item from the release, no network."""

import pytest

from helm.benchmark.metrics.zbench_grader import grade
from helm.benchmark.scenarios.zbench_scenario import ZBenchIVFScenario, item_from_record, parse_layers

RECORD = {
    "id": "IVF-V2-L2-TEST",
    "layer": 2,
    "year": 2021,
    "question": "According to CDC's 2021 Final ART Success Rates data, what percentage of intended egg retrievals "
    "resulted in live-birth deliveries for patients aged 35 to 37 using their own eggs at the clinic? Give the figure "
    "exactly as CDC publishes it and the sample size behind it; if CDC publishes no rate, say exactly what CDC does "
    "publish for that cell.",
    "answer_key": "7/14 (50.0%, small sample)",
    "rules": {
        "pass_all": ["7\\s*/\\s*14"],
        "half_any": ["50(\\.0)?\\s*%"],
        "fail_any": ["not reported"],
        "fail_hard": [],
    },
    "form": "fraction",
    "probe": "encoding: small-sample fraction",
    "cells": [{"y": 2021, "b": "35-37", "m": "intended retrievals to live births", "v": "7/14", "n": 55}],
}


def test_instances_and_grader():
    sc = ZBenchIVFScenario(condition="record")
    instances = sc.instances_from_items([item_from_record(RECORD)])
    assert len(instances) == 1
    assert "CDC record for this clinic" in instances[0].input.text
    assert instances[0].references[0].output.text == "7/14 (50.0%, small sample)"
    extra = instances[0].extra_data
    assert extra is not None
    item = extra["item"]
    assert (
        grade(item, "CDC prints 7/14, a small sample: 7 live births from 14 intended retrievals.")["verdict"] == "Pass"
    )
    assert grade(item, "Not reported for this band.")["verdict"] == "Fail"


def test_layer_filter():
    sc = ZBenchIVFScenario(condition="closed", layers="1")
    assert sc.instances_from_items([item_from_record(RECORD)]) == []
    assert (
        len(ZBenchIVFScenario(condition="closed", layers="1+2").instances_from_items([item_from_record(RECORD)])) == 1
    )
    assert len(ZBenchIVFScenario(condition="closed", layers=2).instances_from_items([item_from_record(RECORD)])) == 1


def test_parse_layers():
    assert parse_layers(None) is None
    assert parse_layers("1+2+6") == {1, 2, 6}
    assert parse_layers(3) == {3}
    with pytest.raises(ValueError):
        parse_layers("none")
