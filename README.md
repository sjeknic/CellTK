[![Documentation Status](https://readthedocs.org/projects/celltk/badge/?version=latest)](https://celltk.readthedocs.io/en/latest/?badge=latest)
[![pytest](https://github.com/sjeknic/CellTK/actions/workflows/main.yml/badge.svg)](https://github.com/sjeknic/CellTK/actions/workflows/main.yml)

![Logo](https://github.com/sjeknic/CellTK/blob/main/docs/logo/black-largeAsset.png)

# CellTK2
Toolkit for analysis of live-cell microscopy data

## Installation

The easiest installation method is via pip:

`pip install celltk2`

To install [BayesianTracker](https://github.com/quantumjot/BayesianTracker) tracking method:

`pip install celltk2[btrack]`

Note: If you are using a Mac with Apple silicon processor, you may run into some issues with this installation. Your best bet may be to try to use conda to install cvxopt before doing the pip install above.

To install [graph-based](https://git.scc.kit.edu/KIT-Sch-GE/2021-cell-tracking) tracking method:

`pip install celltk2[kit]`

If you run into any problems, please open an issue.

## Usage

More details on getting started coming soon. In the meantime, please see the documentation below.

## Documentation
[https://celltk.readthedocs.io/en/latest/](https://celltk.readthedocs.io/en/latest/)

## Credits
This would not be possible without the original CellTK: [https://github.com/braysia/CellTK](https://github.com/braysia/CellTK)