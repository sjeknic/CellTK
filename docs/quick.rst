Quickstart Guide
================

Important classes
------------------

#. ``Pipeline`` - runs CellTK on a single folder of images.
#. ``Orchestrator`` - runs CellTK on multiple folders of images.
#. ``Operation`` - includes ``Segmenter``, ``Processor``, ``Tracker`` and ``Extractor``. Each of these holds the functions for analyzing images and can be found in ``CellTK/celltk``.


Setting up a Pipeline
---------------------

First, initialize a ``Pipeline`` and pass it the folder to your images.

| ``import celltk``
| ``pipe = celltk.Pipeline(parent_folder='/path/to/image/folder')``

Next, build a set of ``Operations`` that you would like to use on those images.