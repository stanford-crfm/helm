"""Tests for codeinsights_code_evaluation_metrics distance semantics.

`Levenshtein.ratio` returns a *similarity* in [0, 1] (1.0 = identical), but the
metric is reported as ``ast_distance`` / ``asm_distance`` and its error
fallback returns 1.0 for "max distance". Computing similarity directly meant a
perfect match was reported as 1.0 — indistinguishable from a parse failure.
These tests pin the distance semantics: identical inputs return 0.0 and a
totally-different input returns 1.0.
"""

import pytest


# The module imports clang.cindex and Levenshtein at top level (both ship in
# the codeinsights extras). Skip cleanly when they're absent.
clang_cindex = pytest.importorskip("clang.cindex")
pytest.importorskip("Levenshtein")


def _patch_clang_index(monkeypatch):
    """Replace clang.cindex.Index.create so ASTAnalyzer.__init__ doesn't need
    libclang installed; ``parse`` round-trips the source text through .cursor
    so the stubbed _extract_ast_features can read it back."""

    class _DummyTU:
        def __init__(self, code: str) -> None:
            self.cursor = code

    class _DummyIndex:
        def parse(self, path, args, unsaved_files):
            return _DummyTU(unsaved_files[0][1])

    monkeypatch.setattr(clang_cindex.Index, "create", staticmethod(lambda: _DummyIndex()))


def test_calculate_ast_distance_identical_returns_zero(monkeypatch):
    _patch_clang_index(monkeypatch)
    from helm.benchmark.metrics.codeinsights_code_evaluation_metrics import ASTAnalyzer

    analyzer = ASTAnalyzer()
    analyzer._extract_ast_features = lambda cursor: list(cursor)

    assert analyzer.calculate_ast_distance("int x = 1;", "int x = 1;") == 0.0


def test_calculate_ast_distance_completely_different_returns_one(monkeypatch):
    _patch_clang_index(monkeypatch)
    from helm.benchmark.metrics.codeinsights_code_evaluation_metrics import ASTAnalyzer

    analyzer = ASTAnalyzer()
    analyzer._extract_ast_features = lambda cursor: list(cursor)

    assert analyzer.calculate_ast_distance("aaaa", "zzzz") == 1.0


def test_calculate_ast_distance_partial_overlap_is_small_positive(monkeypatch):
    _patch_clang_index(monkeypatch)
    from helm.benchmark.metrics.codeinsights_code_evaluation_metrics import ASTAnalyzer

    analyzer = ASTAnalyzer()
    analyzer._extract_ast_features = lambda cursor: list(cursor)

    distance = analyzer.calculate_ast_distance("int x = 1;", "int y = 2;")
    assert 0.0 < distance < 1.0
