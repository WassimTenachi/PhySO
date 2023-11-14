import setuptools
import os

VERSION = '1.1.0a'
DESCRIPTION = 'Physical Symbolic Optimization'

#REQUIRED = open("requirements_pip.txt").read().splitlines()

#EXTRAS = {
#    "display": [
#        "pygraphviz",
#    ],
#}
#EXTRAS['all'] = list(set([item for group in EXTRAS.values() for item in group]))

PATH_FEYNMAN_CSVs = os.path.join("benchmark", "FeynmanDataset", "*.csv")

setuptools.setup(
    name             = 'physo',
    version          = VERSION,
    description      = DESCRIPTION,
    author           = 'Wassim Tenachi',
    author_email     = 'w.tenachi@gmail.com',
    license          = 'MIT',
    packages         = setuptools.find_packages(),
    package_data     = {"physo": [PATH_FEYNMAN_CSVs,]},
    #install_requires = REQUIRED,
    #extras_require   = EXTRAS,
)