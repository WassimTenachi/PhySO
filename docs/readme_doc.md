# Generating documentation

## Requirements

```
pip install sphinx
pip install sphinx-rtd-theme
pip install --upgrade myst-parser
pip install sphinx-math-dollar
pip install sphinxcontrib-video
pip install sphinx_mdinclude
```

## Building documentation

Delete previous build if necessary:
```
rm -r build
```
Building:
```
make html
```
Opening in browser:
```
open build/html/index.html
```

## Building repo readme

From repo root:

```
bash ./docs/make_repo_readme.sh
```