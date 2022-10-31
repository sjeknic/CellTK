.. CellTK documentation master file, created by
   sphinx-quickstart on Thu Mar 17 11:24:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo/black-largeAsset.png

Welcome to CellTK2!
===================

CellTK is collection of tools to simplify working with live-cell and fixed-cell microscopy data. It contains tools for segmenting, tracking, and analyzing images of both mammalian cells and bacterial microcolonies. Once the data are extracted, CellTK further includes, among others, tools for filtering data, building plots, and finding peaks. These tools can be used as stand-alone functions or as part of a larger analysis pipeline. More tools are on the way, and if there is anything you would like to see added, please create an issue on github.


Installation
------------
The easiest way to install CellTK is to use pip.

| ``pip install celltk2``

To install BayesianTracker_

| ``pip install celltk2[btrack]``

Note: If you are using a Mac with Apple silicon processor, you may run into some issues with this installation. Your best bet may be to try to use conda to install cvxopt before doing the pip install above.

To install graph-based tracking_ method:

| ``pip install celltk2[kit]``

If you run into any problems, please open an issue.


Acknowledgments
---------------
This would not be possible without the original CellTK_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick
   pipes
   opers
   arrays
   plots
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _BayesianTracker: https://github.com/quantumjot/BayesianTracker
.. _tracking: https://git.scc.kit.edu/KIT-Sch-GE/2021-cell-tracking
.. _CellTK: https://github.com/braysia/CellTK