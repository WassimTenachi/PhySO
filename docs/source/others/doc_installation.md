# Installation

The package has been tested on:
- Linux
- OSX (ARM & Intel)
- Windows

## Install procedure

### Virtual environment

To install the package it is recommended to first create a conda virtual environment:
```
conda create -n PhySO python=3.12
```
Supporting Python 3.8 - 3.12; other versions may also work.

And activate it:
```
conda activate PhySO
```
### Downloading

`physo` can be downloaded using git:
```
git clone https://github.com/WassimTenachi/PhySO
```

Or by direct downloading a zip of the repository: [here](https://github.com/WassimTenachi/PhySO/zipball/master)

### Dependencies
From the repository root:

Installing dependencies :
```
conda install --file requirements.txt
```

In order to simplify the installation process, since its first version, `physo` has been updated to have minimal very standard dependencies.

---

**NOTE** : `physo` supports CUDA but it should be noted that since the bottleneck of the code is free constant optimization, using CUDA (even on a very high-end GPU) does not improve performances over a CPU and tends to actually hinder performances.

---

### Installing PhySO

Installing `physo` to the environment (from the repository root):
```
python -m pip install -e .
```

### Testing install

Import test:
```
python3
>>> import physo
```
This should result in `physo` being successfully imported.

Unit tests:

Running all unit tests except parallel mode ones (from the repository root):
```
python -m unittest discover -p "*UnitTest.py"
```
This should result in all tests being successfully passed.

Running all unit tests (from the repository root):
```
python -m unittest discover -p "*Test.py"
```
This  should take 5-15 min depending on your system (as if you have a lot of CPU cores, it will take longer to make the efficiency curves).

## Uninstalling
Uninstalling the package.
```
conda deactivate
conda env remove -n PhySO
```

## (Compute Canada users)

If you are using `physo` on [Compute Canada](https://alliancecan.ca/en), follow the steps below.

First, connect to the cluster (e.g., Narval, Beluga) and load one of the modules available for Python (```module avail python```), e.g.:
```
module load python/3.10.13
```
Then, create and activate a virtual environment for `physo`:
```
virtualenv --no-download PhySO
source PhySO/bin/activate
```
Install the required dependencies:
```
pip install --no-index --upgrade pip
pip install torch
pip install sympy
pip install matplotlib
pip install numpy
pip install tqdm
pip install pandas
pip install scikit-learn # For Feynman benchmark analysis script and for density in monitoring plots
pip install jupyterlab # For running the notebooks
```
Finally, download and install `physo`:
```
git clone https://github.com/WassimTenachi/PhySO
python -m pip install -e .
```