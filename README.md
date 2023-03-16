
# $\Phi$-SO : Physical Symbolic Optimization

The physical symbolic regression ( $\Phi$-SO ) package `physo` is a symbolic regression package that fully leverages physical units constraints. For more details see: [[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192).

https://user-images.githubusercontent.com/63928316/225642347-a07127da-a84e-4af3-96f4-4c7fef5a673b.mp4


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
Installing optional dependencies (for advanced debugging in tree representation) :
```
conda install --file requirements_display1.txt
```
```
pip install -r requirements_display2.txt
```

#####  Side note regarding CUDA acceleration:

$\Phi$-SO supports CUDA but it should be noted that since the bottleneck of the code is free constant optimization, using CUDA (even on a very high-end GPU) does not improve performances over a CPU and can actually hinder performances.

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
This should result in all tests being successfully passed (except for program_display_UnitTest tests if optional dependencies were not installed). 

# Getting started

### Symbolic regression with default hyperparameters
[Coming soon] 
In the meantime you can have a look at our demo folder ! :)
### Symbolic regression
[Coming soon]
### Custom symbolic optimization task
[Coming soon]
### Using custom functions
[Coming soon]
### Open training loop
[Coming soon]

# About performances

The main performance bottleneck of `physo` is free constant optimization, therefore, performances are almost linearly dependent on the number of free constant optimization steps and on the number of trial expressions per epoch (ie. the batch size).

In addition, it should be noted that generating monitoring plots takes ~3s, therefore we suggest making monitoring plots every >10 epochs for low time / epoch cases. 

Summary of expected performances with `physo`:

| Time / epoch | Batch size | # free const | free const <br>opti steps | Example                             | Device                                      |
|--------------|------------|--------------|---------------------------|-------------------------------------|---------------------------------------------|
| ~20s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Mac M1 <br>RAM: 16 Go                  |
| ~30s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go   |
| ~250s        | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | GPU: Nvidia GV100 <br>VRAM : 32 Go          |
| ~3s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Mac M1 <br>RAM: 16 Go                  |
| ~3s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go   |
| ~4s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | GPU: Nvidia GV100 <br>VRAM : 32 Go          |

Please note that using a CPU typically results in higher performances than when using a GPU.

# Uninstalling
Uninstalling the package.
```
conda deactivate
conda env remove -n PhySO
```

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
