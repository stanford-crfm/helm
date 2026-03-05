# MedHELM documentation (Jekyll)

This folder is the source for [medhelm.org](https://medhelm.org). The site is built with **Jekyll** and published via **GitHub Pages** (repository setting: use **main** branch, **/docs** folder).

## Local build

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
