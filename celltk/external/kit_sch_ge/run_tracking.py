from pathlib import Path

import numpy as np
from tifffile import imread

from tracker.export import ExportResults
from tracker.extract_data import get_img_files
from tracker.extract_data import get_indices_pandas
from tracker.tracking import TrackingConfig, MultiCellTracker


def run_tracker(img_path, segm_path, res_path, delta_t=3, default_roi_size=2):
    img_path = Path(img_path)
    segm_path = Path(segm_path)
    res_path = Path(res_path)
    img_files = get_img_files(img_path)
    segm_files = get_img_files(segm_path, 'mask')

    # set roi size
    # assume img shape z,x,y
    dummy = np.squeeze(imread(segm_files[max(segm_files.keys())]))
    img_shape = dummy.shape
    masks = get_indices_pandas(imread(segm_files[max(segm_files.keys())]))
    m_shape = np.stack(masks.apply(lambda x: np.max(np.array(x), axis=-1) - np.min(np.array(x), axis=-1) +1))

    if len(img_shape) == 2:
        if len(masks) > 10:
            m_size = np.median(np.stack(m_shape)).astype(int)

            roi_size = tuple([m_size*default_roi_size, m_size*default_roi_size])
        else:
            roi_size = tuple((np.array(dummy.shape) // 10).astype(int))
    else:
        roi_size = tuple((np.median(np.stack(m_shape), axis=0) * default_roi_size).astype(int))

    config = TrackingConfig(img_files, segm_files, roi_size, delta_t=delta_t, cut_off_distance=None)
    tracker = MultiCellTracker(config)
    tracks = tracker()

    exporter = ExportResults()
    exporter(tracks, res_path, tracker.img_shape, time_steps=sorted(img_files.keys()))


if __name__ == '__main__':
    from argparse import ArgumentParser

    PARSER = ArgumentParser(description='Tracking KIT-Sch-GE.')
    PARSER.add_argument('--image_path', type=str, help='path to the folder containing the raw images.')
    PARSER.add_argument('--segmentation_path', type=str, help='path to the folder containing the segmentation images.')
    PARSER.add_argument('--results_path', type=str, help='path where to store the tracking results. '
                                                         'If the results path is the same as the segmentation'
                                                         '_path the segmentation images will be overwritten.')
    PARSER.add_argument('--delta_t', type=int, default=3)
    PARSER.add_argument('--default_roi_size', type=int, default=2)

    ARGS = PARSER.parse_args()

    run_tracker(ARGS.image_path, ARGS.segmentation_path, ARGS.results_path, ARGS.delta_t, ARGS.default_roi_size)
