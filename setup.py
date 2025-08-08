import setuptools
from pathlib import Path
import os
import re

def get_long_description():
    here = Path(__file__).parent
    return (here / "README.md").read_text(encoding="utf-8")

def parse_requirements():
    reqs_path = Path(__file__).parent / "requirements.txt"
    reqs = [
        line.strip()
        for line in reqs_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(('#', '-'))
    ]
    # Replace pytorch by torch (pip name for PyTorch)
    reqs = [req.replace('pytorch', 'torch') for req in reqs]
    return reqs

def get_version():
    version_file = Path(__file__).parent / "physo" / "_version.py"
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file.read_text(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = get_version()
DESCRIPTION = 'Physical Symbolic Optimization'

PATH_FEYNMAN_CSVs = os.path.join("benchmark", "FeynmanDataset", "*.csv")
PATH_CLASS_CSVs   = os.path.join("benchmark", "ClassDataset"  , "*.csv")
package_data = [PATH_FEYNMAN_CSVs, PATH_CLASS_CSVs]

setuptools.setup(
    name             = 'physo',
    version          = VERSION,
    description      = DESCRIPTION,
    long_description              = get_long_description(),
    long_description_content_type ='text/markdown',
    author           = 'Wassim Tenachi',
    author_email     = 'w.tenachi@gmail.com',
    license          = 'MIT',
    packages         = setuptools.find_packages(exclude=['tests*', 'Test.py']),
    package_data         = {"physo": package_data},
    include_package_data = True,
    # Requirements
    python_requires  = '>=3.8',
    install_requires = parse_requirements(),
    # URLs
    url = 'https://github.com/WassimTenachi/PhySO',
    project_urls={
            'Source': 'https://github.com/WassimTenachi/PhySO',
            'Documentation': 'https://physo.readthedocs.io/',
            "Bug Reports": "https://github.com/WassimTenachi/PhySO/issues",
        },
    # Keywords
    classifiers=
    [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
    ],
    keywords = ['physical symbolic optimization',
                'symbolic optimization',
                'machine learning',
                'reinforcement learning',
                'deep learning',
                'physics',
                'symbolic regression',
                'equation discovery',],
)

