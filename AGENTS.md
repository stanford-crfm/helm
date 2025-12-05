# AGENT Instructions

## Development environment
- Use **Python 3.10+**. Create and activate a virtual environment via `virtualenv`, Conda, or `pyenv` before installing dependencies.【F:docs/developer_setup.md†L3-L52】
- Install editable dependencies for development with `pip install --force-reinstall -e .[dev]`; this provides pytest, xdoctest, mypy, flake8, and black pins. A bundled `install-dev.sh` installs pinned requirements (including CUDA-enabled PyTorch on Linux) and then runs `pip install -e .[all,dev]`.【F:docs/developer_setup.md†L54-L107】【F:install-dev.sh†L3-L16】【F:pyproject.toml†L24-L135】【F:pyproject.toml†L422-L434】
- Core CLI entry points are `helm-run`, `helm-summarize`, `helm-server`, and `helm-create-plots`; editable installs expose these commands from `pyproject.toml` script definitions.【F:docs/developer_setup.md†L109-L118】【F:pyproject.toml†L445-L451】
- Optional extras (e.g., `[openai]`, `[metrics]`, `[vlm]`, `[heim]`) enable model- or scenario-specific dependencies; `[all]` aggregates most extras while omitting a few incompatible sets.【F:pyproject.toml†L62-L409】
- Credentials for external providers live in `prod_env/credentials.conf` by default (HOCON). Populate provider keys (OpenAI, Anthropic, Cohere, Google, etc.); additional authentication may be required for Google Cloud (`gcloud auth application-default`) or Hugging Face (`huggingface-cli login`).【F:docs/credentials.md†L3-L47】

## Repository layout
- `src/helm/benchmark/`: pipelines for running evaluations—run specs (`run_specs/`), execution (`run.py`, `runner.py`, `run_spec_factory.py`), registries (`config_registry.py`, `runner_config_registry.py`, `tokenizer_config_registry.py`, `model_metadata_registry.py`, `model_deployment_registry.py`), augmentation and preprocessing helpers, presentation/server assets, and static leaderboard builds.
- `src/helm/benchmark/scenarios/`: scenario definitions for datasets/tasks; `metrics/` for metric implementations; `window_services/` and `tokenizers/` for tokenization and context window utilities.
- `src/helm/clients/`: model client implementations (OpenAI, Anthropic, Together, Google, Hugging Face, etc.), auto-routing, and associated tests/utilities.
- `src/helm/common/`: shared utilities (caching, request/response types, media handling, concurrency, auth/credential helpers, GPU utilities, optional dependency guards, logging helpers).
- `src/helm/config/`: bundled YAML registries describing model deployments, metadata, and tokenizer configs consumed by registries at startup.
- `src/helm/proxy/`: proxy server and CLI for routing model requests.
- `helm-frontend/`: alternative React/TypeScript UI built with Vite/Tailwind; use `yarn install`, `yarn dev`, `yarn test`, `yarn build`, `yarn lint`, `yarn format`. The production/static leaderboard lives under `src/helm/benchmark/static*` rather than here.【F:helm-frontend/README.md†L1-L57】
- `docs/`: MkDocs site with guides (installation, developer setup, adding models/scenarios/tokenizers, credentials, metrics, reproducing leaderboards, HEIM/VHELM/MedHELM/AudioLM docs, etc.).
- Scripts: `install-dev.sh` for Python setup; `pre-commit.sh` for lint/type checks; `install-heim-extras.sh` and `install-shelm-extras.sh` for optional domain extras.

## Key concepts and workflows
- **Running evaluations:** `helm-run` consumes run entries/run specs (under `src/helm/benchmark/run_specs/` and generated via `run_spec_factory.py`), executes scenarios and metrics through `runner.py`, and logs outputs for summarization.
- **Summaries and server:** `helm-summarize` aggregates results; `helm-server` serves the local UI from `helm.benchmark.server` plus static assets in `src/helm/benchmark/static*`.
- **Registries/configs:** `conftest.py` calls `register_builtin_configs_from_helm_package()` before tests to register packaged configs. YAML files in `src/helm/config/` define model/tokenizer metadata consumed by registry modules so new deployments typically require YAML updates plus client/metadata wiring.【F:conftest.py†L1-L5】
- **Model clients:** Extend from `clients/client.py` or auto-routing helpers; many clients rely on provider-specific optional extras. Keep sensitive API calls guarded and configurable via credentials.
- **Scenarios and metrics:** Scenarios live under `benchmark/scenarios/` with dataset loading and instance generation; metrics under `benchmark/metrics/` evaluate responses. Many scenarios/metrics depend on optional extras (`scenarios`, `metrics`, `images`, etc.).

## Testing practices
- Python tests run with `python -m pytest`; default `addopts` skip `models` and `scenarios` markers and enable xdoctest. Use `-m models` or `-m scenarios` to include expensive/networked suites. Verbose mode via `-vv` and targeted file paths encouraged during development.【F:docs/developer_setup.md†L62-L80】【F:pyproject.toml†L485-L509】
- Register built-in configs automatically via `conftest.py`; ensure new tests import registry logic or rely on pytest startup for setup.【F:conftest.py†L1-L5】
- Linting/type checks: run `./pre-commit.sh` or individual `black`, `flake8`, `mypy` commands over `src` and `scripts`. Install git hooks with `pre-commit install` to enforce on push.【F:docs/developer_setup.md†L82-L107】
- Frontend tests use `yarn test`; lint/format with `yarn lint` and `yarn format`.【F:helm-frontend/README.md†L27-L57】

## Extending or modifying
- Follow docs in `docs/adding_new_models.md`, `docs/developer_adding_new_models.md`, `docs/adding_new_scenarios.md`, and `docs/adding_new_tokenizers.md` when introducing models, scenarios, or tokenizers; ensure registry entries and optional dependencies are updated.
- For new model deployments, add YAML entries in `src/helm/config/`, implement/extend clients, and update registries. Guard provider interactions with credentials and optional dependency checks (`helm.common.optional_dependencies`).
- For new scenarios/metrics, place implementations under `benchmark/scenarios/` or `benchmark/metrics/`, add tests, and ensure run specs reference them. Avoid hardcoding credentials/paths; prefer `--local-path` overrides and `prod_env` config directories.
- Maintain formatting (Black 120-char lines) and typing expectations. Use existing dataclasses/utilities (e.g., request/response objects in `helm.common`) rather than ad-hoc structures.

## Documentation pointers
- MkDocs site (root `mkdocs.yml`) indexes guides such as installation, tutorial, run entries, scenarios, metrics, reproduction of leaderboards, and domain-specific docs (VHELM, HEIM, MedHELM, AudioLM, enterprise benchmarks). Quick-start commands in `README.md` are mirrored in `docs/quick_start.md`.
- Developer practices (environment, testing, linting, contributing workflow) are outlined in `docs/developer_setup.md`. Credentials and provider setup live in `docs/credentials.md`.

## Operational notes
- Default local config path is `./prod_env/`; override with `--local-path` when running commands. Ensure required provider credentials are configured before executing model-dependent runs.【F:docs/credentials.md†L3-L47】
- Many scenarios/model clients download datasets or call external APIs; prefer running without `-m models`/`-m scenarios` in CI to avoid costs and failures. Use markers deliberately to target specific expensive suites.【F:pyproject.toml†L485-L509】
- Static leaderboard assets reside under `src/helm/benchmark/static` and `static_build`; React frontend is an alternative UI and not the deployed default.【F:helm-frontend/README.md†L1-L6】
