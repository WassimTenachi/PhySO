# Generating documentation

Documentation can be found [here](https://physo.readthedocs.io/en/latest/).\
Read the docs automatically builds the documentation on each push to the main branch.\
It can be generated locally using the following instructions.

## Requirements

```
python -m pip install -r requirements.txt
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