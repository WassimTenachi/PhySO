# Building PhySO for PyPI

Build can be found [here](https://pypi.org/project/physo/).
It can be built locally using the following instructions (executed from the root of the repository):

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

# Building PhySO for anaconda.org/user channel

Build can be hosted on a [user conda channel](https://anaconda.org/WassimTenachi/physo) enabling the installation using:
```bash
conda install wassimtenachi::physo
```

Instructions to build and upload to a user conda channel (executed from the root of the repository):

## Requirements

```bash
conda install conda-build anaconda-client --yes
```

## Building recipe

Delete previous build if necessary:
```bash
rm -rf ./build/conda-user-recipe/
conda build purge-all
```

Making recipe from latest PyPI package:
```bash
conda skeleton pypi physo --output-dir ./build/conda-user-recipe
```

Patching recipe content:
```bash
./build/patch-conda-user-recipe.sh ./build/conda-user-recipe/physo/meta.yaml
```

## Building conda package

Building the conda package locally:
```bash
conda build ./build/conda-user-recipe/physo
```

Package can now be installed from local conda channel:
```bash
conda install --use-local physo
```

## Uploading to anaconda.org/user channel

Enabling automated upload during building:
```bash
conda config --set anaconda_upload yes 
```
Building the conda package (now triggering an upload to the user channel):
```bash
conda build ./build/conda-user-recipe/physo
```

# Building PhySO for anaconda.org/conda-forge channel

Build can be hosted on the [conda-forge channel](https://anaconda.org/conda-forge/physo) enabling the installation using:
```bash
conda install -c conda-forge physo
```

Instructions to build and upload to the conda-forge channel:

## Initial setup

1. Forked the [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) repository.
2. Added the recipe shown in conda-forge-initial-recipe to the `recipes` folder of your forked repository.
3. Created a pull request to the main conda-forge/staged-recipes repository with title "Add physo recipe".
4. The pull request is then reviewed and merged by the conda-forge maintainers.
 
`physo`'s [pull request](https://github.com/conda-forge/staged-recipes/pull/30733) was approved and merged on 2025-08-07.

A feedstock repository named [conda-forge/physo-feedstock](https://github.com/conda-forge/physo-feedstock/) was automatically created to enable future updates.

## Updating the code

1. A bot detects new releases on PyPI (within 2 hours) and creates a pull request to update the version in the feedstock repository.
2. The pull request is reviewed and merged by `physo`'s maintainers.
3. CDN (Content Delivery Network) propagation across world servers takes ~15 minutes.

## Updating the recipe

1. Fork the [conda-forge/physo-feedstock](https://github.com/conda-forge/physo-feedstock/) repository and make a new branch for the changes.
2. Make the necessary changes to the recipe in the `recipe` folder.
3. Locally build the conda package to ensure the recipe works as expected:
   ```bash
   conda build ./recipe
   ```
4. Create a pull request to the main [conda-forge/physo-feedstock](https://github.com/conda-forge/physo-feedstock/) repository.
5. The pull request is then reviewed and merged by the conda-forge maintainers.

