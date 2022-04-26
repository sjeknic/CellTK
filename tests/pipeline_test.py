import os
import sys

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import numpy as np
import imageio as iio
import pytest

import celltk as ctk


class TestPipeline():
    _orch_path = os.path.join(par, 'examples')
    _img_path = os.path.join(_orch_path, 'live_cell_example')
    _output_path = os.path.join(_img_path, '_output')
    _nuc_weight_path = os.path.join(par, 'celltk/config/unet_example_cell_weights.tf')

    def _make_operations(self):

        pro = ctk.Processor(images=['channel000'], force_rerun=True)
        pro.add_function_to_operation('unet_predict', weight_path=self._nuc_weight_path,
                                      batch=2, save_as='unet')

        seg = ctk.Segmenter(images=['unet'], output='seg', force_rerun=True)
        seg.add_function_to_operation('constant_thres', thres=1.)
        seg.add_function_to_operation('agglomeration_segmentation', agglom_min=0.67,
                                       steps=40, connectivity=1)
        seg.add_function_to_operation('mask_to_image')
        seg.add_function_to_operation('sitk_label')
        seg.add_function_to_operation('filter_objects_by_props',
                                      properties=['area', 'solidity'],
                                      limits=[(25, 100), (0.85, 1.)])

        tra = ctk.Tracker(images=['channel000'], masks=['seg'], output='nuc', force_rerun=True)
        tra.add_function_to_operation('linear_tracker_w_properties',
                                      properties=['centroid', 'total_intensity', 'area'],
                                      weights=[1, 1, 1], mass_thres=0.2, displacement_thres=20)
        tra.add_function_to_operation('detect_cell_division')

        ex = ctk.Extractor(images=['channel000', 'channel001'], tracks=['nuc'],
                            channels=['tritc', 'fitc'], regions=['nuc'], force_rerun=True,
                            time=10, remove_parent=True, min_trace_length=5)
        ex.add_derived_metric('median_ratio',
                              keys=(['nuc', 'fitc', 'median_intensity'],
                                    ['nuc', 'tritc', 'median_intensity']),
                              func='divide', inverse=True, propagate=True)
        ex.add_derived_metric('initial_intensity',
                              keys=(['nuc', 'tritc', 'median_intensity'],),
                              func='nanmean', inverse=False, propagate=False,
                              frame_rng=3, keepdims=True)
        ex.add_derived_metric('final_intensity',
                              keys=(['nuc', 'tritc', 'median_intensity'],),
                              func='nanmean', inverse=False, propagate=False,
                              frame_rng=-3, keepdims=True)
        ex.add_derived_metric('initial_ratio',
                              keys=(['nuc', 'fitc', 'median_intensity'],
                                    ['nuc', 'tritc', 'initial_intensity']),
                              func='divide', inverse=False, propagate=True)
        ex.add_derived_metric('final_ratio',
                              keys=(['nuc', 'fitc', 'median_intensity'],
                                    ['nuc', 'tritc', 'final_intensity']),
                              func='divide', inverse=False, propagate=True)
        ex.add_derived_metric('convex_area_ratio',
                              keys=(['nuc', 'tritc', 'area'],
                                    ['nuc', 'tritc', 'convex_area']),
                              func='divide', inverse=False, propagate=True)
        ex.add_derived_metric('predict_peaks',
                                keys=(('nuc', 'fitc', 'median_intensity'),),
                                propagate='true', function='predict_peaks')
        ex.add_derived_metric('active',
                                keys=(('nuc', 'fitc', 'peaks'),),
                                propagate='true', function='peaks')

        return [pro, seg, tra, ex]


    def test_pipeline(self):

        # Test making pipeline and operations
        pipe = ctk.Pipeline(parent_folder=self._img_path,
                            output_folder=self._output_path,
                            skip_frames=(2,))
        ops = self._make_operations()
        pipe.add_operations(ops)
        assert len(pipe.operations) == 4

        # Test saving as yaml files
        pipe.save_as_yaml(fname='_pipe.yaml')
        pipe.save_operations_as_yaml(fname='_ops.yaml')

        # Reload from yaml to test it was saved properly
        pipe = ctk.Pipeline.load_from_yaml(os.path.join(self._output_path, '_pipe.yaml'))

        # Test running
        pipe.run()

        # Check for outputs
        cond_arr = ctk.ConditionArray.load(os.path.join(self._output_path, 'data_frame.hdf5'))

        # Check that frame was skipped
        test_arr = iio.imread(os.path.join(self._output_path, 'nuc', 'track2.tiff'))
        assert (test_arr == 0).all()

    def test_orchestrator(self):

        # Test making Orchestrator and operations
        orch = ctk.Orchestrator(parent_folder=self._orch_path,
                                output_folder=self._output_path,
                                match_str='live_cell', condition_map={'live_cell_example': 'example'},
                                skip_frames=(2,))  # Test skipping frames too
        ops = self._make_operations()
        orch.add_operations(ops)
        assert len(orch.operations) == 4
        assert len(orch.pipelines) == 1

        # Test saving as yaml files
        orch.save_pipelines_as_yamls()
        orch.save_operations_as_yaml()
        orch.save_condition_map_as_yaml()

        # Test running
        orch.run()

        # Test building files
        orch.build_experiment_file()

        # Check for outputs
        cond_arr = ctk.ConditionArray.load(os.path.join(self._output_path, 'live_cell_example',
                                                        'data_frame.hdf5'))
        exp_arr = ctk.ExperimentArray.load(os.path.join(self._output_path,
                                                        'experiment.hdf5'))

        # Check that the frame was skipped
        test_arr = iio.imread(os.path.join(self._output_path, 'live_cell_example', 'nuc', 'track2.tiff'))
        assert (test_arr == 0).all()
