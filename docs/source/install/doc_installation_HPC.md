## HPC & Compute Canada users

If you are using `physo` on an HPC such as [Compute Canada](https://alliancecan.ca/en), follow the steps below.

First, connect to the cluster (e.g., Narval, Beluga) and load one of the modules available for Python (```module avail python```) on your HPC, e.g.:
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
pip install torch >= 1.11.0
pip install numpy
pip install sympy
pip install pandas
pip install matplotlib
pip install scikit-learn
```
Finally, download and install `physo`:
```
git clone https://github.com/WassimTenachi/PhySO
python -m pip install -e . --no-deps
```
