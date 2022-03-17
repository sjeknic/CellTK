.. CellTK documentation master file, created by
   sphinx-quickstart on Thu Mar 17 11:24:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CellTK!
==================

CellTK is collection of tools to simplify working with live-cell and fixed-cell microscopy data. It contains tools for segmenting, tracking, and analyzing images of both mammalian cells and bacterial microcolonies. Once the data are extracted, CellTK further includes, among others, tools for filtering data, building plots, finding peaks, and calculating mutual information. These tools can be used as stand-alone functions or as part of a larger analysis pipeline. More tools are on the way, and if there is anything you would like to see added, please create an issue on github.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick
   opers
   arrays

Installation
------------
There are a couple dependency conflicts that I am still resolving. Until those are resolved, the package will not be published to Pypi. Please use the following steps to
install and run CellTK.

| ``git clone https://github.com/sjeknic/CellTK/``
| ``pip install -r requirements.txt``
| ``pip install -r nodep_requirements.txt --no-deps``

That's it! You should be good to go. If you run into any problems, please open an issue on github.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
