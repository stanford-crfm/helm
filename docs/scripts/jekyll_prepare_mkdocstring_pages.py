#!/usr/bin/env python3
"""Expand MkDocs-only docs/*.md pages for Jekyll (GitHub Pages).

Reference pages may use MkDocstrings (``::: …``) or mkdocs-macros Jinja, which
Jekyll does not process. We run ``mkdocs build`` once, extract each page's
Material article HTML, convert to Markdown, and overwrite the corresponding
``docs/*.md`` before ``jekyll build``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# (site subdirectory under site/, Jekyll front-matter title, docs markdown filename)
def _escape_liquid_braces(text: str) -> str:
    """Avoid Jekyll/Liquid parsing ``{{``/``}}`` from docstrings (e.g. BibTeX, prompts)."""
    return text.replace("{{", "&#123;&#123;").replace("}}", "&#125;&#125;")


PAGES: list[tuple[str, str, str]] = [
    ("models", "Models", "models.md"),
    ("metrics", "Metrics", "metrics.md"),
    ("scenarios", "Scenarios", "scenarios.md"),
    ("perturbations", "Perturbations", "perturbations.md"),
    ("schemas", "Schemas", "schemas.md"),
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "docs"

    try:
        from bs4 import BeautifulSoup
        import html2text
    except ImportError as err:
        print("Install: pip install beautifulsoup4 html2text", file=sys.stderr)
        raise SystemExit(1) from err

    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "-q"],
        cwd=repo_root,
        check=True,
    )

    h = html2text.HTML2Text()
    h.body_width = 0

    for site_subdir, title, md_name in PAGES:
        built = repo_root / "site" / site_subdir / "index.html"
        if not built.is_file():
            raise SystemExit(f"Missing MkDocs output: {built}")

        html = built.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        article = soup.select_one("article.md-content__inner")
        if article is None:
            raise SystemExit(f"Could not find article in {built}")

        body = _escape_liquid_braces(h.handle(str(article)).strip()) + "\n"
        out_path = docs_dir / md_name
        out_path.write_text(f"---\ntitle: {title}\n---\n\n{body}", encoding="utf-8")
        print(f"Wrote Jekyll-ready {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
