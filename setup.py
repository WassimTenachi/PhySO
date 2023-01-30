import setuptools

VERSION = '1.0dev'
DESCRIPTION = 'Physical Symbolic Regression'

#REQUIRED = open("requirements_pip.txt").read().splitlines()

#EXTRAS = {
#    "display": [
#        "pygraphviz",
#    ],
#}
#EXTRAS['all'] = list(set([item for group in EXTRAS.values() for item in group]))


setuptools.setup(
    name             = 'physr',
    version          = VERSION,
    description      = DESCRIPTION,
    author           = 'Wassim Tenachi',
    author_email     = 'w.tenachi@gmail.com',
    license          = 'MIT',
    packages         = setuptools.find_packages(),
    #install_requires = REQUIRED,
    #extras_require   = EXTRAS,
)