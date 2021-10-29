"""Minimal examples to test full tracking pipeline."""
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import disk, ball
from tifffile import imread
from tifffile import imsave

from evaluation.tracking_metrics import calc_tra_score
from tracker.export import ExportResults
from tracker.extract_data import get_img_files
from tracker.extract_data import get_indices_pandas
from tracker.tracking import TrackingConfig, MultiCellTracker

np.random.seed(42)


def run_tracking_pipeline(img_path, segm_path, res_path, delta_t):
    """Runs the tracking pipeline based on the test data set"""
    img_path = Path(img_path)
    img_files = get_img_files(img_path)
    segm_files = get_img_files(segm_path)

    # set roi size
    # assume img shape z,x,y
    dummy = np.squeeze(imread(segm_files[max(segm_files.keys())]))
    img_size = dummy.shape
    masks = get_indices_pandas(imread(segm_files[max(segm_files.keys())]))
    m_shape = np.stack(masks.apply(lambda x: np.max(np.array(x), axis=-1) - np.min(np.array(x), axis=-1) + 1))

    if len(img_size) == 2:
        m_size = np.median(np.stack(m_shape)).astype(int)
        roi_size = tuple([m_size * 2, m_size * 2])
    else:
        roi_size = tuple((np.median(np.stack(m_shape), axis=0) * 2).astype(int))

    config = TrackingConfig(img_files, segm_files, roi_size, delta_t=delta_t, cut_off_distance=None)
    tracker = MultiCellTracker(config)
    tracks = tracker()
    for t_id, track in tracks.items():
        print(t_id, track.pred_track_id, track.successors)
    exporter = ExportResults()
    exporter(tracks, res_path, tracker.img_shape, time_steps=sorted(img_files.keys()))

#################
# Check 2D      #
#################


def test_multi_merged_2d():
    """Simulates an under-segmentation error on one time point."""
    # t_0: 5 objects, t_1: 1 object, t_2: 5 objects segmented
    # tracking goal: 5 tracked objects
    path_2d = Path(__file__).parent / 'test_2D'
    data_path = path_2d / 'multi_merge'
    disk_size = 17
    ground_truth_img = np.zeros((120, 120))
    img_sizeize = np.array(ground_truth_img.shape).reshape(-1, 1) - 1
    disk_positions = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30)]
    t_steps = 3
    t_merged = 1
    for i, obj_center in enumerate(disk_positions):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_sizeize] = np.tile(img_sizeize, (1, mask.shape[-1]))[mask > img_sizeize]
        ground_truth_img[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    binary_img = (ground_truth_img > 0).astype(np.int)
    for i in range(t_steps):
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), ground_truth_img.astype(np.uint16))
        if i > 0 and t_merged - i >= 0:
            segmentation_img = (ground_truth_img > 0).astype(np.uint16)
        else:
            segmentation_img = ground_truth_img
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))

    gt_lineage = pd.DataFrame.from_dict({(i+1): [0, t_steps - 1, 0] for i in range(len(disk_positions))}).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_2d)


def test_multi_split_2D():
    """Simulates an over-segmentation error on one time point."""
    # t_0: 1 objects, t_1: 5 objects, t_2: 1 object segmented
    # tracking goal: 1 tracked object
    path_2d = Path(__file__).parent / 'test_2D'
    data_path = path_2d / 'multi_split'
    disk_size = 17
    ground_truth_img = np.zeros((120, 120))
    img_sizeize = np.array(ground_truth_img.shape).reshape(-1, 1) - 1
    disk_positions = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30)]
    t_steps = 3
    t_merged = 2
    for i, obj_center in enumerate(disk_positions):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_sizeize] = np.tile(img_sizeize, (1, mask.shape[-1]))[mask > img_sizeize]
        ground_truth_img[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    binary_img = (ground_truth_img > 0).astype(np.int)
    for i in range(t_steps):
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        gt_img = ground_truth_img > 0
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), gt_img.astype(np.uint16))
        if i > 0 and t_merged - i >= 0:
            segmentation_img = (ground_truth_img > 0).astype(np.uint16)
        else:
            segmentation_img = ground_truth_img
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))

    gt_lineage = pd.DataFrame.from_dict({1: [0, t_steps - 1, 0]}).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_2d)


def test_skip_2D():
    """Models missing segmentation mask for a time point."""
    # t_0: 5 objects, t_1: 4 objects (one missing), t_2: 5 objects segmented
    # tracking goal: 5 tracked objects
    path_2d = Path(__file__).parent / 'test_2D'
    data_path = path_2d / 'multi_skip'
    disk_size = 17
    ground_truth_img = np.zeros((120, 120))
    img_sizeize = np.array(ground_truth_img.shape).reshape(-1, 1) - 1
    disk_positions = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30)]
    t_steps = 3
    t_skip = 1
    for i, obj_center in enumerate(disk_positions):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_sizeize] = np.tile(img_sizeize, (1, mask.shape[-1]))[mask > img_sizeize]
        ground_truth_img[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    binary_img = (ground_truth_img > 0).astype(np.int)
    for i in range(t_steps):
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), ground_truth_img.astype(np.uint16))
        if i > 0 and t_skip - i >= 0:
            segmentation_img = ground_truth_img.copy().astype(np.uint16)
            segmentation_img[segmentation_img == 2] = 0
        else:
            segmentation_img = ground_truth_img
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))

    gt_lineage = pd.DataFrame.from_dict({(i+1): [0, t_steps - 1, 0] for i in range(len(disk_positions))}).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=2)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_2d)


def test_mitosis_2D():
    """Simulates cell division."""
    # t_0: 5 objects, t_1: 5 objects, t_2: 6 objects segmented (mitosis occurred)
    # tracking goal: 5 tracked objects at t_0,t_1,
    # 6 at t_2 with correctly assigned mother-daughter relationship
    path_2d = Path(__file__).parent / 'test_2D'
    data_path = path_2d / 'mitosis'
    disk_size = 17
    pre_mitosis_gt = np.zeros((120, 120))
    post_mitosis_gt = np.zeros((120, 120))
    img_size = np.array(pre_mitosis_gt.shape).reshape(-1, 1) - 1
    disk_positions_pre_mitosis = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30)]
    disk_positions_post_mitosis = [(30, 60), (60, 60),
                                   (90, 60), (30, 30), (60, 30),
                                   (50, 50), (70, 72)]
    t_steps = 3
    t_mitosis = 2
    for i, obj_center in enumerate(disk_positions_pre_mitosis):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        pre_mitosis_gt[tuple(mask)] = (i+1)

    for i, obj_center in enumerate(disk_positions_post_mitosis):
        if i == 1:
            continue
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        post_mitosis_gt[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    for i in range(t_steps):
        if i < t_mitosis:
            binary_img = (pre_mitosis_gt > 0).astype(np.int)
            segmentation_img = pre_mitosis_gt
        else:
            binary_img = (post_mitosis_gt > 0).astype(np.int)
            segmentation_img = post_mitosis_gt
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
    lineage = {1: [0, 2, 0],
               2: [0, 1, 0],
               3: [0, 2, 0],
               4: [0, 2, 0],
               5: [0, 2, 0],
               6: [2, 2, 2],
               7: [2, 2, 2]}
    gt_lineage = pd.DataFrame.from_dict(lineage).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_2d)


def test_appear_2D():
    """Simulates appearing object."""
    # t_0: 5 objects, t_1: 6 objects (object appeared), t_2: 6 objects segmented
    # tracking goal: 5 tracked objects at t_0,  6 at t_1, t_2
    path_2d = Path(__file__).parent / 'test_2D'
    data_path = path_2d / 'appear'
    disk_size = 17
    gt_pre_appearance = np.zeros((120, 120))
    gt_post_appearance = np.zeros((120, 120))
    img_size = np.array(gt_pre_appearance.shape).reshape(-1, 1) - 1
    disk_positions_pre_appearance = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30)]
    disk_positions_post_appearance = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30),
                                      (10, 20)]
    t_steps = 3
    t_appear = 1
    for i, obj_center in enumerate(disk_positions_pre_appearance):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_pre_appearance[tuple(mask)] = (i + 1)

    for i, obj_center in enumerate(disk_positions_post_appearance):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_post_appearance[tuple(mask)] = (i + 1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    for i in range(t_steps):
        if i < t_appear:
            binary_img = (gt_pre_appearance > 0).astype(np.int)
            segmentation_img = gt_pre_appearance
        else:
            binary_img = (gt_post_appearance > 0).astype(np.int)
            segmentation_img = gt_post_appearance
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
    lineage = {1: [0, 2, 0],
               2: [0, 2, 0],
               3: [0, 2, 0],
               4: [0, 2, 0],
               5: [0, 2, 0],
               6: [1, 2, 0],
               }
    gt_lineage = pd.DataFrame.from_dict(lineage).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_2d)


def test_delete_2D():
    """Simulates disappearing object."""
    # t_0: 6 objects, t_1: 5 objects (object disappeared), t_2: 5 objects segmented
    # tracking goal: 6 tracked objects at t_0,
    # 5 at t_1, t_2 with correctly detected disappearing object
    path_2d = Path(__file__).parent / 'test_2D'
    data_path = path_2d / 'delete'
    disk_size = 17
    gt_pre_deletion = np.zeros((120, 120))
    gt_post_deletion = np.zeros((120, 120))
    img_size = np.array(gt_pre_deletion.shape).reshape(-1, 1) - 1
    disk_positions_post_deletion = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30)]
    disk_positions_pre_deletion = [(30, 60), (60, 60), (90, 60), (30, 30), (60, 30),
                                   (10, 20)]
    t_steps = 3
    t_delete = 1
    for i, obj_center in enumerate(disk_positions_post_deletion):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_post_deletion[tuple(mask)] = (i + 1)

    for i, obj_center in enumerate(disk_positions_pre_deletion):
        indices = np.where(disk(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_pre_deletion[tuple(mask)] = (i + 1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    for i in range(t_steps):
        if i < t_delete:
            binary_img = (gt_pre_deletion > 0).astype(np.int)
            segmentation_img = gt_pre_deletion
        else:
            binary_img = (gt_post_deletion > 0).astype(np.int)
            segmentation_img = gt_post_deletion
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
    lineage = {1: [0, 2, 0],
               2: [0, 2, 0],
               3: [0, 2, 0],
               4: [0, 2, 0],
               5: [0, 2, 0],
               6: [0, 0, 0],
               }
    gt_lineage = pd.DataFrame.from_dict(lineage).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_2d)

#################
# Check 3D      #
#################

def test_multi_merged_3D():
    """Simulates an under-segmentation error at one time point."""
    # t_0: 5 objects, t_1: 1 object, t_2: 5 objects segmented
    # tracking goal: 5 tracked objects
    path_3d = Path(__file__).parent / 'test_3D'
    data_path = path_3d / 'multi_merge'
    disk_size = 17
    ground_truth_img = np.zeros((120, 120, 120))
    img_size = np.array(ground_truth_img.shape).reshape(-1, 1) - 1
    disk_positions = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50)]
    t_steps = 3
    t_merged = 1
    for i, obj_center in enumerate(disk_positions):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        ground_truth_img[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path/'01')
        os.makedirs(data_path/'01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    binary_img = (ground_truth_img > 0).astype(np.int)
    for i in range(t_steps):
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path/'01'/('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path/'01_GT' / 'TRA'/('man_track' + str(i).zfill(3) + '.tif'), ground_truth_img.astype(np.uint16))
        if i > 0 and t_merged - i >= 0:
            segmentation_img = (ground_truth_img > 0).astype(np.uint16)
        else:
            segmentation_img = ground_truth_img
        imsave(data_path/'01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))

    gt_lineage = pd.DataFrame.from_dict({(i+1): [0, t_steps - 1, 0] for i in range(len(disk_positions))}).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_3d)


def test_multi_split_3D():
    """Simulates an over-segmentation error at one time point."""
    # t_0: 1 objects, t_1: 5 objects, t_2: 1 object segmented
    # tracking goal: 1 tracked object
    path_3d = Path(__file__).parent / 'test_3D'
    data_path = path_3d / 'multi_split'
    disk_size = 17
    ground_truth_img = np.zeros((120, 120, 120))
    img_size = np.array(ground_truth_img.shape).reshape(-1, 1) - 1
    disk_positions = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50)]
    t_steps = 3
    t_merged = 2
    for i, obj_center in enumerate(disk_positions):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        ground_truth_img[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    binary_img = (ground_truth_img > 0).astype(np.int)
    for i in range(t_steps):
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        gt_img = ground_truth_img > 0
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), gt_img.astype(np.uint16))
        if i > 0 and t_merged - i >= 0:
            segmentation_img = (ground_truth_img > 0).astype(np.uint16)
        else:
            segmentation_img = ground_truth_img
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))

    gt_lineage = pd.DataFrame.from_dict({1: [0, t_steps - 1, 0]}).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_3d)


def test_skip_3D():
    """Simulates a missing segmentation mask at one time point."""
    # t_0: 5 objects, t_1: 4 objects (one missing), t_2: 5 objects segmented
    # tracking goal: 5 tracked objects
    path_3d = Path(__file__).parent / 'test_3D'
    data_path = path_3d / 'multi_skip'
    disk_size = 17
    ground_truth_img = np.zeros((120, 120, 120))
    img_size = np.array(ground_truth_img.shape).reshape(-1, 1) - 1
    disk_positions = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50)]
    t_steps = 3
    t_skip = 1
    for i, obj_center in enumerate(disk_positions):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        ground_truth_img[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    binary_img = (ground_truth_img > 0).astype(np.int)
    for i in range(t_steps):
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), ground_truth_img.astype(np.uint16))
        if i > 0 and t_skip - i >= 0:
            segmentation_img = ground_truth_img.copy().astype(np.uint16)
            segmentation_img[segmentation_img == 2] = 0
        else:
            segmentation_img = ground_truth_img
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))

    gt_lineage = pd.DataFrame.from_dict({(i+1): [0, t_steps - 1, 0] for i in range(len(disk_positions))}).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=2)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_3d)


def test_mitosis_3D():
    """Simulates cell division."""
    # t_0: 5 objects, t_1: 5 objects, t_2: 6 objects segmented (mitosis occurred)
    # tracking goal: 5 tracked objects at t_0,t_1, 6 at t_2
    path_3d = Path(__file__).parent / 'test_3D'
    data_path = path_3d / 'mitosis'
    disk_size = 17
    pre_mitosis_gt = np.zeros((120, 120, 120))
    post_mitosis_gt = np.zeros((120, 120, 120))
    img_size = np.array(pre_mitosis_gt.shape).reshape(-1, 1) - 1
    disk_positions_pre_mitosis = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50)]
    disk_positions_post_mitosis = [(30, 60, 50), (60, 60, 50),
                                   (90, 60, 50), (30, 30, 50), (60, 30, 50),
                                   (50, 50, 40), (70, 72, 60)]
    t_steps = 3
    t_mitosis = 2
    for i, obj_center in enumerate(disk_positions_pre_mitosis):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        pre_mitosis_gt[tuple(mask)] = (i+1)

    for i, obj_center in enumerate(disk_positions_post_mitosis):
        if i == 1:
            continue
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_center).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        post_mitosis_gt[tuple(mask)] = (i+1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    for i in range(t_steps):
        if i < t_mitosis:
            binary_img = (pre_mitosis_gt > 0).astype(np.int)
            segmentation_img = pre_mitosis_gt
        else:
            binary_img = (post_mitosis_gt > 0).astype(np.int)
            segmentation_img = post_mitosis_gt
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
    lineage = {1: [0, 2, 0],
               2: [0, 1, 0],
               3: [0, 2, 0],
               4: [0, 2, 0],
               5: [0, 2, 0],
               6: [2, 2, 2],
               7: [2, 2, 2]}
    gt_lineage = pd.DataFrame.from_dict(lineage).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_3d)


def test_appear_3D():
    """Simulates an appearing object."""
    # tracking goal: 5 tracked objects at t_0,t_1,
    # 6 at t_2 with correctly assigned mother-daughter relationship
    path_3d = Path(__file__).parent / 'test_3D'
    data_path = path_3d / 'appear'
    disk_size = 17
    gt_pre_appearance = np.zeros((120, 120, 120))
    gt_post_appearance = np.zeros((120, 120, 120))
    img_size = np.array(gt_pre_appearance.shape).reshape(-1, 1) - 1
    disk_positions_pre_appearance = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50)]
    disk_positions_post_appearance = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50),
                                      (10, 20, 50)]
    t_steps = 3
    t_appear = 1
    for i, obj_size in enumerate(disk_positions_pre_appearance):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_size).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_pre_appearance[tuple(mask)] = (i + 1)

    for i, obj_size in enumerate(disk_positions_post_appearance):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_size).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_post_appearance[tuple(mask)] = (i + 1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    for i in range(t_steps):
        if i < t_appear:
            binary_img = (gt_pre_appearance > 0).astype(np.int)
            segmentation_img = gt_pre_appearance
        else:
            binary_img = (gt_post_appearance > 0).astype(np.int)
            segmentation_img = gt_post_appearance
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
    lineage = {1: [0, 2, 0],
               2: [0, 2, 0],
               3: [0, 2, 0],
               4: [0, 2, 0],
               5: [0, 2, 0],
               6: [1, 2, 0],
               }
    gt_lineage = pd.DataFrame.from_dict(lineage).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_3d)


def test_delete_3D():
    """Simulates a disappearing object."""
    # t_0: 6 objects, t_1: 5 objects (object disappeared), t_2: 5 objects segmented
    # tracking goal: 6 tracked objects at t_0,
    # 5 at t_1, t_2 with correctly detected disappearing object
    path_3d = Path(__file__).parent / 'test_3D'
    data_path = path_3d / 'delete'
    disk_size = 17
    gt_pre_deletion = np.zeros((120, 120, 120))
    gt_post_deletion = np.zeros((120, 120, 120))
    img_size = np.array(gt_pre_deletion.shape).reshape(-1, 1) - 1
    disk_positions_post_deletion = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50)]
    disk_positions_pre_deletion = [(30, 60, 50), (60, 60, 50), (90, 60, 50), (30, 30, 50), (60, 30, 50),
                                   (10, 20, 50)]
    t_steps = 3
    t_delete = 1
    for i, obj_size in enumerate(disk_positions_post_deletion):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_size).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_post_deletion[tuple(mask)] = (i + 1)

    for i, obj_size in enumerate(disk_positions_pre_deletion):
        indices = np.where(ball(disk_size) > 0)
        mask = np.array(indices) - disk_size + np.array(obj_size).reshape(-1, 1)
        mask[mask < 0] = 0
        mask[mask > img_size] = np.tile(img_size, (1, mask.shape[-1]))[mask > img_size]
        gt_pre_deletion[tuple(mask)] = (i + 1)

    try:
        os.makedirs(data_path / '01')
        os.makedirs(data_path / '01_GT' / 'TRA')
        os.makedirs(data_path / '01_SEG')
    except FileExistsError:
        print(f'Skip creating directory {data_path}')

    # save img, masks
    for i in range(t_steps):
        if i < t_delete:
            binary_img = (gt_pre_deletion > 0).astype(np.int)
            segmentation_img = gt_pre_deletion
        else:
            binary_img = (gt_post_deletion > 0).astype(np.int)
            segmentation_img = gt_post_deletion
        img = binary_img + np.random.randn(*binary_img.shape) * 0.005
        img = (img * 15000).astype(np.uint16)
        imsave(data_path / '01' / ('t' + str(i).zfill(3) + '.tif'), img)
        imsave(data_path / '01_GT' / 'TRA' / ('man_track' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
        imsave(data_path / '01_SEG' / ('mask' + str(i).zfill(3) + '.tif'), segmentation_img.astype(np.uint16))
    lineage = {1: [0, 2, 0],
               2: [0, 2, 0],
               3: [0, 2, 0],
               4: [0, 2, 0],
               5: [0, 2, 0],
               6: [0, 0, 0],
               }
    gt_lineage = pd.DataFrame.from_dict(lineage).T
    gt_lineage.to_csv((data_path / '01_GT' / 'TRA' / 'man_track.txt').as_posix(), sep=' ', header=None)

    run_tracking_pipeline(data_path / '01', data_path / '01_SEG', data_path / '01_RES', delta_t=1)
    result = calc_tra_score(data_path / '01_RES', data_path / '01_GT')
    assert np.abs(result[0] - 1) < 1e-12, f'tracking score and assumed score missmatch {result[0]}'
    shutil.rmtree(path_3d)


if __name__ == '__main__':
    test_skip_2D()



