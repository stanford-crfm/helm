# Writing Documentation

Our documentation relies on `mkdocs`.

## Installation

Before you can install the packages required for the documentation, you need to follow the the [developer setup](developer_setup.md) which will install a local copy of `crfm-helm`.

Then, install the documentation dependencies with:
```bash
pip install -r docs/requirements.txt
```

You can then build the documentation with:
```bash
mkdocs serve
```
And visit localhost:8000 to see the documentation.

## Adding new documentation

To add new documentation, simply add a new markdown file to the `docs/` folder. You can then link to it from other markdown files using the following syntax:
```markdown
[Link text](path/to/file.md)
```
If you want it to appear in the sidebar, you will need to add it to `mkdocs.yml` under `nav`.
Subtitles will automatically be added to the sidebar if you use the markdown header syntax (e.g. `## Subtitle`).

Everything under **Reference** (Models, Metrics, Perturbations, ...) do not need to be manually updated. They are automatically generated from the code.