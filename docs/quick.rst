Quickstart Guide
================

Important classes
------------------

#. ``Pipeline`` - runs CellTK on a single folder of images.
#. ``Orchestrator`` - runs CellTK on multiple folders of images.
#. ``Operation`` - includes ``Segmenter``, ``Processor``, ``Tracker``, ``Extractor``, and ``Evaluator``. Each of these holds the functions for analyzing images and can be found in ``CellTK/celltk``.


Setting up a Pipeline
---------------------

To use, save all of the following in a python script and run it. First, initialize a ``Pipeline`` and pass it the folder to your images. For this example, we will use the images in the example folder. Optionally, you can also pass a path to ``output_folder``. By default, outputs will be stored in a new directory called ``outputs``.

::

    import celltk
    pipe = celltk.Pipeline(parent_folder='/your/path/to/CellTK/examples/live_cell_example')


Next, build a set of ``Operations`` that you would like to use on those images. We first need to initialize the operation and let it know which images to use by passing a unique string that is in the name of those image files. In this case the nuclear channel can be identified with "channel000". We tell it that the output should be called "nuc". We also add a few extra options to save our final output and skip functions if it finds the files are already made.

::

    seg = celltk.Segmenter(images=['channel000'], output='seg', save=True, force_rerun=False)

Next, we add the functions to the operation. For this example, we want to use UNet to find the nuclei followed by a simple constant threshold and cleaning to label the nuclei. Any kwargs those functions require can be passed to ``add_function_to_operation``. We will use the example weights for UNet, but if you have your own weights, you can pass them with the kwarg ``weight_path``. For any function, you can add the kwarg ``save_as`` to save those output files. It's best to add this to time consuming operations so that they do not need to be repeated.

::

    seg.add_function_to_operation('unet_predict', save_as='unet_nuc')
    seg.add_function_to_operation('constant_thres', thres=0.8)
    seg.add_function_to_operation('clean_labels', min_radius=3, relabel=True)

Next, we will add a tracking operation using the same format as above. This time we import the output from ``seg`` to be used by the ``Tracker``. We will use the ``kit_sch_ge_tracker`` function which runs a tracking algorithm from `Katharina Loeffler and colleagues`_.

::

    tra = celltk.Tracker(images=['channel000'], masks='seg', output='nuc',
                         save=True, force_rerun=False)
    tra.add_function_to_operation('kit_sch_ge_tracker')

Finally, we need to add an operation to extract the data and save it in an easy to use file. For this, we use ``Extractor``. This is the operation to pass most of the experimental metadata to. No functions are added to this operation.

::

    ext = celltk.Extractor(images=['channel000', 'channel0001'], tracks=['nuc'],
                           regions=['nuc'], channels=['tritc', 'fitc'],
                           time=10, min_trace_length=5, force_rerun=True)

Now we are ready to run everything! For this, we simply add the operations to the pipeline and hit go.

::

    pipe.add_operations([seg, tra, ext])
    pipe.run()

This will create the output folder, run all the operations, and save a file called ``data_frame.hdf5`` with all of the data saved as a ``ConditionArray``.


Setting up an Orchestrator
--------------------------

Orchestrator es a very similar API as ``Pipeline``. ``parent_folder`` should point to a directory that contains sub-directories of images.


Utilizing extracted data
------------------------

After the pipeline runs, the data will be saved in an ``hdf5`` file. To access these data, load the file as a ``ConditionArray``. For this example, we will use ``examples/example_df.hdf5``.

::

    array = celltk.ConditionArray.load('examples/example_df.hdf5')
    print(array.shape)
    > (1, 2, 24, 42, 6)

All ``ConditionArrays`` are five-dimensional. The dimensions are regions (e.g. nucleus, cytoplasm), channels (e.g. TRITC, FITC), metrics (e.g. median_intensity, area), cells, and frames. The first three dimensions can be indexed using strings, while the last two dimensions are indexed using integers. Currently, every indexing operation on a ``ConditionArray`` returns an ``np.ndarray``. Addtionally, the array will always be at least two dimensional.

::

    data = array['nuc', 'fitc', 'area']
    print(data.shape)
    > (42, 6)
    print(type(data))
    > numpy.ndarray

You can also index multiple items in each axis using a ``list`` or ``tuple``. For example, you may want to get the ``x`` and ``y`` positions of each cell.

::

    position = array['nuc', 'fitc', ('x', 'y')]
    print(data.shape)
    > (2, 42, 6)



.. _Katharina Loeffler and colleagues: https://git.scc.kit.edu/KIT-Sch-GE/2021-cell-tracking