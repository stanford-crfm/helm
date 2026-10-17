"""zBench IVF: record-grounded healthcare facility reasoning (zReports HealthCare Data Lab).

5,144 tasks generated from CDC's clinic-level ART success-rate tables (2020 to 2022); every answer key is a cell of
the public
record with a row locator that re-derives it from the published file; grading is deterministic (see zbench_metrics.py).
Benchmark page: https://api.zreports.org/lab/zbench. Dataset: https://huggingface.co/datasets/shawnzreports/zbench-ivf.
MedHELM category: Administration and Workflow (facility records, reporting periods, measures), not clinical decision
support.

Conditions:
  closed  the question alone
  record  the clinic's own CDC cells printed under the question (the control for evidence access)
Layers: `layers=1+2+6` keeps those layers only (HELM run entries split on commas, so `+` separates the numbers).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Union

from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference, Scenario
from helm.common.general import ensure_directory_exists

HF_REPO = "shawnzreports/zbench-ivf"
HF_REVISION = "03a2611888784cd7de9f6463235dc01bdabc6c5b"  # the dataset commit that completed release zbench-ivf-1.2
RELEASE = "zbench-ivf-1.2"
RECORD_PREFIX = (
    "Answer from the CDC record supplied below. Give the figure exactly as CDC publishes it, with its qualifier "
    "(small sample, suppressed, no cycles) when the record carries one; do not substitute another year, age band or "
    "measure.\n\n"
)


def item_from_record(rec: Dict[str, Any], neighbours: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A public.jsonl record (plus its neighbours) as the item the grader expects."""
    return {
        "id": rec["id"],
        "layer": rec.get("layer"),
        "band": rec.get("band"),
        "clinic": rec.get("clinic"),
        "year": rec.get("year"),
        "question": rec["question"],
        "answer_key": rec.get("answer_key") or "",
        "rubric": rec.get("rubric") or {},
        "rules": rec.get("rules") or {},
        "verify": rec.get("verify"),
        "meta": {
            "form": rec.get("form"),
            "forms": rec.get("forms"),
            "cells": rec.get("cells") or [],
            "neighbours": list(neighbours or []),
            "probe": rec.get("probe"),
            "template": rec.get("template"),
            "measure": rec.get("measure"),
            "type": rec.get("type"),
            "trap": rec.get("trap"),
        },
    }


def record_excerpt(item: Dict[str, Any]) -> str:
    """The clinic's own cells, printed the way the reference harness's record tools print them."""
    lines = []
    for c in item["meta"].get("cells") or []:
        where = ", ".join(str(x) for x in (c.get("y"), c.get("b"), c.get("m")) if x is not None)
        extra = " (all-ages count %s)" % c["n"] if c.get("n") not in (None, "") else ""
        lines.append("- %s: %s%s" % (where, c.get("v"), extra))
    return "CDC record for this clinic (as published):\n" + "\n".join(lines) if lines else ""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_items(public_path: str, neighbours_path: Optional[str] = None) -> List[Dict[str, Any]]:
    nb: Dict[str, List[Dict[str, Any]]] = {}
    if neighbours_path and os.path.exists(neighbours_path):
        for rec in read_jsonl(neighbours_path):
            nb[rec["id"]] = rec.get("neighbours") or []
    return [item_from_record(rec, nb.get(rec["id"])) for rec in read_jsonl(public_path)]


def parse_layers(layers: Optional[Union[str, int]]) -> Optional[Set[int]]:
    """`None` keeps every layer; otherwise the layer numbers in `layers` (`1+2+6`; a lone number arrives as an int)."""
    if layers is None or layers == "":
        return None
    found = {int(x) for x in re.findall(r"\d+", str(layers))}
    if not found:
        raise ValueError("layers must be layer numbers separated by '+', for example layers=1+2+6")
    return found


class ZBenchIVFScenario(Scenario):
    """zBench IVF: questions generated from CDC's clinic-level ART tables; every answer key is a CDC row."""

    name = "zbench_ivf"
    description = (
        "Record-grounded healthcare facility questions (CDC ART clinic tables 2020 to 2022) with deterministic, "
        "row-provenanced grading."
    )
    tags = ["healthcare", "facility_records", "question_answering", "administration_and_workflow"]

    def __init__(self, condition: str = "closed", layers: Optional[Union[str, int]] = None):
        super().__init__()
        if condition not in ("closed", "record"):
            raise ValueError("condition must be 'closed' or 'record'")
        self.condition = condition
        self.layers = parse_layers(layers)

    def _download(self, output_path: str) -> Dict[str, Optional[str]]:
        """The release's public and neighbours files, pinned to the dataset revision, cached under output_path."""
        from huggingface_hub import hf_hub_download  # a dependency of `datasets`, which HELM already requires

        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        public = hf_hub_download(
            HF_REPO, RELEASE + "-public.jsonl", repo_type="dataset", revision=HF_REVISION, cache_dir=data_path
        )
        try:
            neighbours: Optional[str] = hf_hub_download(
                HF_REPO, RELEASE + "-neighbours.jsonl", repo_type="dataset", revision=HF_REVISION, cache_dir=data_path
            )
        except Exception:  # optional: without it the grader runs but does not label substitutions
            neighbours = None
        return {"public": public, "neighbours": neighbours}

    def instances_from_items(self, items: List[Dict[str, Any]]) -> List[Instance]:
        out: List[Instance] = []
        for it in items:
            if self.layers is not None and int(it.get("layer") or 0) not in self.layers:
                continue
            text = it["question"]
            if self.condition == "record":
                text = RECORD_PREFIX + text + "\n\n" + record_excerpt(it)
            out.append(
                Instance(
                    input=Input(text=text),
                    references=[Reference(Output(text=it.get("answer_key") or ""), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                    id=it["id"],
                    extra_data={
                        "item": it,
                        "layer": it.get("layer"),
                        "probe": it["meta"].get("probe"),
                        "condition": self.condition,
                    },
                )
            )
        return out

    def get_instances(self, output_path: str) -> List[Instance]:
        paths = self._download(output_path)
        return self.instances_from_items(load_items(paths["public"] or "", paths["neighbours"]))
