# Installation

The package has been tested on:
- Linux
- OSX (ARM & Intel)
- Windows

## Install procedure

If you are encountering issues with the installation, please refer to the [FAQ](https://physo.readthedocs.io/en/latest/r_faq.html) or raise an issue on the [GitHub repository](https://github.com/WassimTenachi/PhySO/issues).

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

In order to simplify the installation process, `physo` has very few and standard dependencies, which are listed in the `requirements.txt` file.

---

**Without Conda (Alternative)** : 

If you are unable to use conda or prefer to use another environment manager, you can install the dependencies manually using `pip`:  
``` bash
pip install --no-index --upgrade pip
pip install torch >= 1.11.0
pip install numpy
pip install sympy
pip install pandas
pip install matplotlib
pip install scikit-learn
```
This approach is especially useful on some HPC systems, see the [HPC installation guide](https://physo.readthedocs.io/en/latest/r_installation.html#hpc-compute-canada-users).

---

---

**NOTE** : `physo` supports CUDA but it should be noted that since the bottleneck of the code is free constant optimization, using CUDA (even on a very high-end GPU) does not improve performances over a CPU and tends to actually hinder performances.

---

### Installing PhySO

Installing `physo` to the environment (from the repository root):
```
python -m pip install -e . --no-deps
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
Uninstalling the package:
```
pip uninstall physo
```
Removing the conda environment:
```
conda deactivate
conda env remove -n PhySO
```
