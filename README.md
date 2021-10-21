This repository contains the code to the publication:  
 A graph-based cell tracking algorithm with few manually tunable parameters and automated segmentation error correction  
Katharina Löffler, Tim Scherr, Ralf Mikut  
bioRxiv 2021.03.16.435631; doi: https://doi.org/10.1101/2021.03.16.435631  

The code has been tested on Windows and Linux using Python 3.8.
## Setup
### 1) create folder structure
- create a project directory LOCAL_DIR
- create two folders named data and results in LOCAL_DIR
- clone the code and install dependencies:
```
conda create --name venv_graph_tracking_kit_sch_ge_2021 python==3.8
conda activate venv_graph_tracking_kit_sch_ge_2021
git clone git@git.scc.kit.edu:KIT-Sch-GE/2021-cell-tracking.git
pip install -e ./2021-cell-tracking
```
so the final structure is  
```
LOCAL_DIR
└───data  (contains the ground truth data sets)
└───2021-cell-tracking (contains our tracking code)
└───results (synthetically degraded segmentation masks will be stored here)
```

### 2) Install gurobi
see help_gurobi.txt

## run tracking
- the tracking algorithm can be used with any 2D/3D image data with a segmentation which needs to be provided by the user
- it is assumed that the image data and segmentation data have a similar naming convention as used by the cell tracking challenge (http://celltrackingchallenge.net)
```
python -m run_tracking --image_path IMAGE_PATH --segmentation_path SEGMENTATION_PATH --results_path RESULTS_PATH

```

## Reproduce synthetic data sets
### 1) download data sets
- go to http://celltrackingchallenge.net
and download the training data sets Fluo-N2DH-SIM+ and Fluo-N3DH-SIM+, unpack and save in data directory:
```
LOCAL_DIR
└───data  (contains the ground truth data sets)
│   └───Fluo-N2DH-SIM+
│   └───Fluo-N3DH-SIM+
└───2021-cell-tracking
└───results (synthetically degraded segmentation masks stored will be stored here)
```

### 2) run code
- run create_synth_segm_data.py to create synthetically degraded segmentation mask images
```
python -m create_synth_segm_data
```

