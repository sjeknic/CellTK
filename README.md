[![Documentation Status](https://readthedocs.org/projects/celltk/badge/?version=docs2)](https://celltk.readthedocs.io/en/docs2/?badge=docs2)
[![pytest](https://github.com/sjeknic/CellTK/actions/workflows/main.yml/badge.svg)](https://github.com/sjeknic/CellTK/actions/workflows/main.yml)


# CellTK
Toolkit for analysis of live-cell microscopy data

## Installation

The standard `pip install -r requirements.txt` will install most, but not all of the required packages. This is due to supposed conflicts in dependency versions, that don't affect function of this repo. A separate file contains packages to install without dependencies.

Proper installation:  
`pip install -r requirements.txt`  
`pip install -r nodep_requirements.txt --no-deps`  

Note: You may get warnings during installation of `mahotas`. This is normal, as long as it installs, you're good to go.

## Documentation
https://celltk.readthedocs.io/en/latest/