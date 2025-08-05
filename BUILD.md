# Building PhySO for PyPI

Build can be found [here](https://pypi.org/project/physo/).
It can be built locally using the following instructions.

## Requirements

```bash
pip install --upgrade build twine
```

## Building package

Delete previous build if necessary:
```bash
rm -r dist
```

Building:
```bash
python -m build
```

## Uploading to Test PyPI

It is recommended to upload to Test PyPI first to ensure everything works as expected.
```bash
twine upload --repository testpypi dist/*
```

Package can be found [here](https://test.pypi.org/project/physo/) and installed using:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple physo
```
`--extra-index-url` is used to allow installation of dependencies from the main PyPI repository.

## Uploading to PyPI

After testing on Test PyPI, the package can be uploaded to the main PyPI repository:
```bash
twine upload dist/*
```
Package can be found [here](https://pypi.org/project/physo/) and installed using:
```bash
pip install physo
```



