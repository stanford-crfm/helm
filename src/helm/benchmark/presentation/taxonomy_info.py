from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaxonomyInfo:
    # Task (e.g., question answering)
    task: Optional[str] = None

    # Domain - genre (e.g., Wikipedia)
    what: Optional[str] = None

    # Domain - when it was written (e.g., 2010s)
    when: Optional[str] = None

    # Domain - demographics (e.g., web users)
    who: Optional[str] = None

    # Language (e.g., English)
    language: Optional[str] = None
