# Installation

The package has been tested on:
- Linux
- OSX (ARM & Intel)
- Windows

## Install procedure

### Virtual environment

To install the package it is recommended to first create a conda virtual environment:
```
conda create -n PhySO python=3.8
```
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

Installing essential dependencies :
```
conda install --file requirements.txt
```
Installing optional dependencies (for advanced debugging in tree representation) :
```
conda install --file requirements_display1.txt
```
```
pip install -r requirements_display2.txt
```

In addition, latex should be installed on the system.

---

**NOTE** : `physo` supports CUDA but it should be noted that since the bottleneck of the code is free constant optimization, using CUDA (even on a very high-end GPU) does not improve performances over a CPU and can actually hinder performances.

---

### Installing PhySO

Installing `physo` (from the repository root):
```
pip install -e .
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
This should result in all tests being successfully passed (except for program_display_UnitTest tests if optional dependencies were not installed).

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
