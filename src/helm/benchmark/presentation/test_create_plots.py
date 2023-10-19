# mypy: check_untyped_defs = False
from helm.common.general import asdict_without_nones
from helm.benchmark.presentation.table import Table, Cell, HeaderCell
from helm.benchmark.presentation.create_plots import parse_table


def test_table_parsing():
    title = "table"
    scenarios = ["A", "B", "C", "D"]
    models = ["X", "Y", "Z"]
    header = []
    rows = [[] for m in models]

    header.append(HeaderCell("Models"))
    header.append(HeaderCell("Mean win rate"))
    for s in scenarios:
        header.append(HeaderCell(s, lower_is_better=True, metadata={"run_group": s, "metric": "accuracy"}))
    for i, model in enumerate(models):
        rows[i].append(Cell(model))
        rows[i].append(Cell(0.1 * i))
        for j, s in enumerate(scenarios):
            rows[i].append(Cell(i * 10 + j))
    summarize_table = Table(title, header, rows)
    table = parse_table(asdict_without_nones(summarize_table))
    assert table.adapters == models
    assert list(table.mean_win_rates) == [0.0, 0.1, 0.2]
    assert len(table.columns) == len(scenarios)
    for j, c in enumerate(table.columns):
        assert c.group == scenarios[j]
        assert c.lower_is_better
        assert c.metric == "accuracy"
        for i, v in enumerate(c.values):
            assert v == i * 10 + j
