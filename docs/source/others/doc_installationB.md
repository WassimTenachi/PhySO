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