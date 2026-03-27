---
title: Editing Documentation
---

# Editing Documentation

The documentation that you are reading now is an invaluable resource for newcomers and experienced users alike. Contributions to the documentation are very welcome.

We use [Jekyll](https://jekyllrb.com/) to build the site and [GitHub Pages](https://pages.github.com/) to host it at [medhelm.org](https://medhelm.org).

To edit the documentation, first clone the repository locally, then install HELM from the repository by following the [Developer Setup](/developer_setup) instructions. To build and preview the site locally, from the repository root run:

```sh
cd docs
bundle install
bundle exec jekyll serve
```

Then open [http://localhost:4000/](http://localhost:4000/) to view the documentation.

The source Markdown files are in the `docs/` folder. Add new pages there and they will appear in the site. The navigation is defined in `_includes/header.html`.
