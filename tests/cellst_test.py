import sys
import os

# Correct import paths
cwd = os.path.dirname(os.path.realpath(__file__))
par = os.path.dirname(cwd)
sys.path.insert(0, par)

import cellst

# Test creating all objects
def test_object_creation(output=False):
    """"""
    #cellst.Pipeline(verbose=True, log_file=False, overwrite=False)
    cellst.Orchestrator(verbose=True, log_file=False, overwrite=False, dry_run=True)
    cellst.Segment(save=False)
    cellst.Process(save=False)
    cellst.Track(save=False)
    cellst.Extract(save=False)
    cellst.Evaluate(save=False)
    cellst.SlurmController()
