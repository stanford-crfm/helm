# MedHELM documentation (Jekyll)

**Live documentation:** [medhelm.org](https://medhelm.org)

This folder is the source for the site. It is built with **Jekyll** and deployed to **gh-pages** via a GitHub Action on push to **main** (when `docs/**` or the workflow file changes). Configure GitHub Pages to serve from the **gh-pages** branch (Settings → Pages → Source: Deploy from a branch → Branch: gh-pages, / (root)).

## MedHELM library (quick reference)

Documentation on [medhelm.org](https://medhelm.org) covers the full workflow. Summary:

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `pip install medhelm` or `uv pip install medhelm` | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** | `pip install "medhelm[summarization]"` | DischargeMe, ACI-Bench, Patient-Edu (install may take 2–3 min) |
| **Gated** | `pip install "medhelm[gated]"` | MedQA, MedMCQA (Google Drive) |

Run a benchmark:

```bash
uv run medhelm-run --run-entries "<scenario>:model=<model>" --suite <name> --max-eval-instances <n>
uv run helm-summarize --suite <name>
uv run helm-server --suite <name>
```

Example (standard): `uv run medhelm-run --run-entries "pubmed_qa:model=huggingface/qwen2.5-7b" --suite my_med_test --max-eval-instances 10`

## Local build (Jekyll)

**Ruby >= 3.0 is required** (Ruby 3.x and Ruby 4.x are supported). macOS ships with Ruby 2.6; install a newer Ruby with Homebrew:

```bash
brew install ruby
# Add to your PATH (e.g. in ~/.zshrc): export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
# Then:
cd docs
bundle install
bundle exec jekyll serve
```

Open http://localhost:4000/

(Alternatively, use [rbenv](https://github.com/rbenv/rbenv) or [asdf](https://asdf-vm.com/) to install Ruby 3 or 4.)

## Virtual environment (Python)

From the repository root, a Python virtual environment is available for project tooling (e.g. running HELM or MkDocs for API reference):

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

Jekyll does not use the Python venv; it uses Bundler (Ruby). 
