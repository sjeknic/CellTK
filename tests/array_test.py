import os
import sys

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import numpy as np
import pytest

from celltk.core.arrays import ConditionArray, ExperimentArray


class TestArray():
    # TODO: Replace with smaller files
    _cond_array_path = os.path.join(par, 'examples', 'example_df.hdf5')
    _exp_array_path = os.path.join(par, 'examples', 'example_experiment.hdf5')
    _cond_temp_path = os.path.join(par, 'examples', '_df.hdf5')
    _exp_temp_path = os.path.join(par, 'examples', '_exp.hdf5')

    def _test_load_array(self):
        # ConditionArray should fail to load ExperimentArray
        with pytest.raises(TypeError):
            ConditionArray.load(self._exp_array_path)
        # Experiment array CAN load a condition array
        # TODO: Make this no longer possible
        ExperimentArray.load(self._cond_array_path)


        # Load the arrays properly
        self.cond_arr = ConditionArray.load(self._cond_array_path)
        self.exp_arr = ExperimentArray.load(self._exp_array_path)

        # Assert that the arrays are the full dimension
        assert all(self.cond_arr.shape)
        assert all([all(s) for s in self.exp_arr.shape])

    def test_filter_condition(self):
        self._test_load_array()

        # Try removing no cells from the array - confirm shape is same
        old_shape = self.cond_arr.shape
        mask = self.cond_arr.remove_short_traces(0)
        self.cond_arr.filter_cells(mask, delete=True)
        assert self.cond_arr.shape == old_shape

        # Try removing all cells
        mask = self.cond_arr.remove_short_traces(np.inf)
        self.cond_arr.filter_cells(mask, delete=True)
        assert self.cond_arr.shape != old_shape
        assert self.cond_arr.coordinate_dimensions['cells'] == 0

        # Try saving and loading the array
        self.cond_arr.save(self._cond_temp_path)
        _cond = ConditionArray.load(self._cond_temp_path)
        assert _cond.shape == self.cond_arr.shape
        assert _cond.keys == self.cond_arr.keys
        assert _cond.condition == self.cond_arr.condition
        assert np.isclose(_cond.time, self.cond_arr.time, equal_nan=True).all()
        assert np.isclose(_cond._arr, self.cond_arr._arr, equal_nan=True).all()

    def test_saving_loading_arrays(self):
        self._test_load_array()

        # Save to temporary files
        self.cond_arr.save(self._cond_temp_path)
        self.exp_arr.save(self._exp_temp_path)

        # Load from those files
        _cond = ConditionArray.load(self._cond_temp_path)
        _exp = ExperimentArray.load(self._exp_array_path)

        # Confirm loading produces the original array
        assert _cond.shape == self.cond_arr.shape
        assert _cond.keys == self.cond_arr.keys
        assert _cond.condition == self.cond_arr.condition
        assert np.isclose(_cond.time, self.cond_arr.time, equal_nan=True).all()
        assert np.isclose(_cond._arr, self.cond_arr._arr, equal_nan=True).all()

        assert _exp.shape == self.exp_arr.shape
        assert _exp.name == self.exp_arr.name
        assert tuple(_exp.keys()) == tuple(self.exp_arr.keys())
        assert ([np.isclose(_t, t, equal_nan=True).all()
                 for _t, t in zip(_exp.time, self.exp_arr.time)])
        assert ([np.isclose(_v._arr, v._arr, equal_nan=True).all()
                 for _v, v in zip(_exp.values(), self.exp_arr.values())])

        # Delete the old files
        os.remove(self._cond_temp_path)
        os.remove(self._exp_temp_path)

        # Confirm they are deleted
        assert not os.path.exists(self._cond_temp_path)
        assert not os.path.exists(self._exp_temp_path)

        # Confirm they cannot be loaded
        with pytest.raises(FileNotFoundError):
            ConditionArray.load(self._cond_temp_path)
            ExperimentArray.load(self._exp_temp_path)
