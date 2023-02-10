
# $\Phi$-SO : Physical Symbolic Regression

The physical symbolic regression ($\Phi$-SR) package `physo` is a symbolic regression package that fully leverages physical units constraints. For more details see: [paper].

# Installation

### Virtual environment

The package has been tested on Unix and OSX. To install the package it is recommend to first create a conda virtual environment:
```
conda create -n PhySO python=3.8
```
And activate it:
```
conda activate PhySO
```
### Dependencies
From the repository root:

Installing essential dependencies :
```
conda install --file requirements.txt
```
Installing optional dependencies (for monitoring plots) :
```
pip install -r requirements_display.txt
```
#####  Side note for ARM users:

The file `requirements_display.txt` contains dependencies that can be installed via pip only. However, it also contains `pygraphviz` which can be installed via conda which avoids compiler issues on ARM. 

It is recommended to run:
```
conda install pygraphviz==1.9
```
before running:
```
pip install -r requirements_display.txt
```
### Installing $\Phi$-SR

Installing `physo` (from the repository root):
```
pip install -e .
```

### Testing install

#####  Import test:
```
python3
>>> import physo
```
This should result in `physo` being successfully imported.

#####  Unit tests:

From the repository root:
```
python -m unittest discover -p "*UnitTest.py"
```
This should result in all tests being successfully passed (except for plots tests if dependencies were not installed). 

# Getting started

### Fitting data
### Custom functions
### Custom symbolic optimization task
### Open training loop


# Citing this work

```
[bibtex of paper]
```
