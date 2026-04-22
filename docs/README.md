# MedHELM documentation (Jekyll)

**Live documentation:** [medhelm.org](https://medhelm.org)

This folder is the source for the site. It is built with **Jekyll** and deployed to **gh-pages** via a GitHub Action on push to **main** (when `docs/**` or the workflow file changes). Configure GitHub Pages to serve from the **gh-pages** branch (Settings → Pages → Source: Deploy from a branch → Branch: gh-pages, / (root)).

## MedHELM library (quick reference)

Documentation on [medhelm.org](https://medhelm.org) covers the full workflow. Summary:

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `uv pip install -e .` (repo) or `pip install medhelm` / `uv pip install medhelm` (PyPI) | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** (Clinical NLP tier) | `pip install "medhelm[summarization]"` | DischargeMe, ACI-Bench, Patient-Edu (install may take 2–3 min; adds bert-score, rouge-score, nltk) |
| **Gated** (licensing tier) | `pip install "medhelm[gated]"` | MedQA, MedMCQA (Google Drive; adds gdown) |

**Quick test** (runs in seconds): `uv run medhelm-run --run-entries "pubmed_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite my_med_test --max-eval-instances 2` then `helm-summarize` and `helm-server`.

**Full example:** `uv run medhelm-run --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite my_med_test --max-eval-instances 10` then `helm-summarize` and `helm-server`.

## Local build (Jekyll)

**Ruby >= 3.0 is required** (Ruby 3.x and Ruby 4.x are supported). macOS ships with Ruby 2.6; install a newer Ruby with Homebrew:

```bash
brew install ruby
# Add to your PATH (e.g. in ~/.zshrc): export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
# Then:
cd docs
bundle install
```

### Full site preview (recommended)

Several pages (`models.md`, `metrics.md`, `scenarios.md`, `perturbations.md`, `schemas.md`) are **MkDocs sources** in git (mkdocstrings / mkdocs-macros). Jekyll’s **Liquid** parser does not understand that syntax, so a plain `bundle exec jekyll serve` fails on `models.md`.

CI runs `docs/scripts/jekyll_prepare_mkdocstring_pages.py` before Jekyll to expand those files. Do the same locally from the **repository root**:

```bash
pip install -r docs/requirements.txt   # once: MkDocs, mkdocstrings, beautifulsoup4, html2text, …
chmod +x docs/scripts/serve_jekyll_local.sh   # once, if needed
./docs/scripts/serve_jekyll_local.sh
```

Open http://localhost:4000/ — when you stop the server (Ctrl+C), the script restores the original `docs/*.md` sources from git so nothing is left expanded by mistake.

### Jekyll only (home and non-MkDocs pages)

If you only need a quick look at layouts like the homepage and do not care that `/models/` and other reference pages are missing or broken, you can still run `cd docs && bundle exec jekyll serve` after temporarily moving aside `models.md`, or rely on the script above for a faithful build.

(Alternatively, use [rbenv](https://github.com/rbenv/rbenv) or [asdf](https://asdf-vm.com/) to install Ruby 3 or 4.)

## Virtual environment (Python)

From the repository root, a Python virtual environment is available for project tooling (e.g. running HELM or MkDocs for API reference):

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

Jekyll does not use the Python venv; it uses Bundler (Ruby).

