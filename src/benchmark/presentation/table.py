from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict


@dataclass(frozen=True)
class Cell:
    # Semantic value (that can be used for sorting)
    value: Any

    # Optionally, if we want to render things specially (floating points to 3 decimal points)
    display_value: Optional[str] = None

    # Detailed description if hover over the cell
    description: Optional[str] = None

    # If we click on the link for this cell, it takes us somewhere
    href: Optional[str] = None

    # Styling
    style: Dict[str, Any] = None


@dataclass(frozen=True)
class Hyperlink:
    text: str
    href: str


@dataclass(frozen=True)
class Table:
    title: str
    header: List[Cell]
    rows: List[List[Cell]]

    # Extra information to show at the bottom
    links: List[Hyperlink] = field(default_factory=list)

    # Optional descrition to show at the top
    description: Optional[str] = None


def table_to_latex(table: Table, table_name: str, skip_blank_columns=True) -> str:
    """Return a string representing the latex version of the table."""
    columns_shown = list(range(len(table.header)))
    if skip_blank_columns:
        columns_shown = [i for i in columns_shown if not all(row[i].value is None for row in table.rows)]

    lines = []
    lines.append("\\begin{table*}[htp]")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{l" + "r" * (len(columns_shown) - 1) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(str(table.header[i].value) for i in columns_shown) + " \\\\")
    lines.append("\\midrule")
    for row in table.rows:
        lines.append(" & ".join(str(row[i].display_value or row[i].value or "") for i in columns_shown) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\caption{Results for " + table_name + "}")
    lines.append("\\label{fig:" + table_name + "}")
    lines.append("\\end{table*}")

    latex = "\n".join(lines).replace("%", "\\%")
    return latex
