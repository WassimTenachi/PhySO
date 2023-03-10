
# $\Phi$-SO : Physical Symbolic Optimization

The physical symbolic regression ( $\Phi$-SO ) package `physo` is a symbolic regression package that fully leverages physical units constraints. For more details see: [[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192).

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
### Installing $\Phi$-SO

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

### Symbolic regression with default hyperparameters
[Coming soon] 
In the meantime you can have a look a our demo folder ! :)
### Symbolic regression
[Coming soon]
### Custom symbolic optimization task
[Coming soon]
### Using custom functions
[Coming soon]
### Open training loop
[Coming soon]


# Citing this work

```
@ARTICLE{2023arXiv230303192T,
       author = {{Tenachi}, Wassim and {Ibata}, Rodrigo and {Diakogiannis}, Foivos I.},
        title = "{Deep symbolic regression for physics guided by units constraints: toward the automated discovery of physical laws}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning, Physics - Computational Physics},
         year = 2023,
        month = mar,
          eid = {arXiv:2303.03192},
        pages = {arXiv:2303.03192},
          doi = {10.48550/arXiv.2303.03192},
archivePrefix = {arXiv},
       eprint = {2303.03192},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230303192T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
