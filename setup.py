import setuptools

VERSION = '1.0dev'
DESCRIPTION = 'Physical Symbolic Optimization'

#REQUIRED = open("requirements_pip.txt").read().splitlines()

#EXTRAS = {
#    "display": [
#        "pygraphviz",
#    ],
#}
#EXTRAS['all'] = list(set([item for group in EXTRAS.values() for item in group]))


setuptools.setup(
    name             = 'physo',
    version          = VERSION,
    description      = DESCRIPTION,
    author           = 'Wassim Tenachi',
    author_email     = 'w.tenachi@gmail.com',
    license          = 'MIT',
    packages         = setuptools.find_packages(),
    #install_requires = REQUIRED,
    #extras_require   = EXTRAS,
)