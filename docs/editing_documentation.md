# Editing Documentation

The documentation that you are reading now is an invaluable resource for newcomers and experienced users alike. Contributions to the documentation are very welcome.

We currently use the [MkDocs](https://www.mkdocs.org/) as our static site generator and [ReadTheDocs](https://readthedocs.org/) as our web host.

To edit the documentation, first clone the repository locally, then install HELM from the repository by following the [Developer Setup](developer_setup.md) instructions. After that, install the MkDocs dependencies by running the following from the root of the repository:

```sh
pip install -r docs/requirements.txt
```

You should now be able to run MkDocs from the root of the repository:

```sh
mkdocs serve
```

Then navigate to [http://localhost:8000/](http://localhost:8000/) to view your locally-built documentation.

The source Markdown files for the documentation are stored in the `docs/` folder. By default, MkDocs watches the source directories for changes and automatically re-renders the web pages when it detects changes.

If you are creating a new page, you should add your page to the `nav` section in `mkdocs.yml`. This will add your page to the table of contents in the side menu.

We make heavy use of plugins and macros for auto-generating documentation from code and docstrings. For more information, please refer to the documentation for these plugins e.g. [mkdocs-macros](https://mkdocs-macros-plugin.readthedocs.io/en/latest/), [mkdocstrings](https://mkdocstrings.github.io/python/) and [mkdocstrings-python](https://mkdocstrings.github.io/python/).
