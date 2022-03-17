Quickstart Guide
================

Important classes
------------------

#. ``Pipeline`` - runs CellTK on a single folder of images.
#. ``Orchestrator`` - runs CellTK on multiple folders of images.
#. ``Operation`` - includes ``Segmenter``, ``Processor``, ``Tracker`` and ``Extractor``. Each of these holds the functions for analyzing images and can be found in ``CellTK/celltk``.


Setting up a Pipeline
---------------------

To use, save all of the following in a python script and run it. First, initialize a ``Pipeline`` and pass it the folder to your images. For this example, we will use the images in the example folder. Optionally, you can also pass a path to ``output_folder``. By default, outputs will be stored in a new directory called ``outputs``.

| ``import celltk``
| ``pipe = celltk.Pipeline(parent_folder='/your/path/to/CellTK/examples/D4-Site_2')``

Next, build a set of ``Operations`` that you would like to use on those images. We fist need to initialize the operation and let it know which images to use by passing a unique string that is in the name of those image files. In this case the nuclear channel can be identified with "channel000". We tell it that the output should be called "nuc". We also add a few extra options to save our final output and skip functions if it finds the files are already made.

| ``seg = celltk.Segmenter(images=['channel000'], output='seg', save=True, force_rerun=False)``

Next, we add the functions to the operation. For this example, we want to use UNet to find the nuclei followed by a simple constant threshold and cleaning to label the nuclei. Any kwargs those functions require can be passed to ``add_function_to_operation``. We will use the example weights for UNet, but if you have your own weights, you can pass them with the kwarg ``weight_path``. For any function, you can add the kwarg ``save_as`` to save those output files. It's best to add this to time consuming operations so that they do not need to be repeated.

| ``seg.add_function_to_operation('unet_predict', save_as='unet_nuc')``
| ``seg.add_function_to_operation('constant_thres', thres=0.8)``
| ``seg.add_function_to_operation('clean_labels', min_radius=3, relabel=True)``

Next, we will add a tracking operation using the same format as above. This time we import the output from ``seg`` to be used by the ``Tracker``. We will use the ``kit_sch_ge_tracker`` function which runs a tracking algorithm from `Katharina Loeffler and colleagues`_.

| ``tra = celltk.Tracker(images=['channel000'], masks='seg', output='nuc', save=True, force_rerun=False)``
| ``tra.add_function_to_operation('kit_sch_ge_tracker')``

Finally, we need to add an operation to extract the data and save it in an easy to use file. For this, we use ``Extractor``. This is the operation to pass most of the experimental metadata to. No functions are added to this operation.

| ``ext = celltk.Extractor(images=['channel000', 'channel0001'], tracks=['nuc'],
                           regions=['nuc'], channels=['tritc', 'fitc']),
                           time=10, min_trace_length=5, force_rerun=True``

Now we are ready to run everything! For this, we simply add the operations to the pipeline and hit go.

| ``pipe.add_operations([seg, tra, ext])``
| ``pipe.run()``

This will create the output folder, run all the operations, and save a file called ``data_frame.hdf5`` with all of the data saved as a ``ConditionArray``.

Setting up an Orchestrator
--------------------------

Uses a very similar API as ``Pipeline``. ``parent_folder`` should point to a directory that contains sub-directories of images.



.. _Katharina Loeffler and colleagues: https://git.scc.kit.edu/KIT-Sch-GE/2021-cell-tracking