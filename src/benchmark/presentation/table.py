from dataclasses import dataclass
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
class Table:
    title: str
    header: List[Cell]
    rows: List[List[Cell]]
