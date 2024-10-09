from typing import List


class ChineseTokenizer:
    """Chinese tokenizer.

    Used by CLEVA for computing metrics on Chinese text."""

    def tokenize(self, text: str) -> List[str]:
        # Tokenize by characters to avoid a dependency on word segmentation methods.
        return [c for c in text]
