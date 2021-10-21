"""Utilities"""
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from tifffile import imread, imsave


def collect_paths(path_to_dir):
    """Returns a list of full paths to the lowest subdirectories of the provided path"""
    folder_content = os.walk(path_to_dir)
    sub_paths = [sub_path[0] for sub_path in folder_content if not sub_path[1]]
    for index in range(len(sub_paths)):
        sub_paths[index] = sub_paths[index].replace('\\', '/')
    return sub_paths


def compress_tifffiles(data_path):
    """
    Compresses tiff files in a folder.
    Args:
        data_path: path to a folder containing tiff files

    """
    data_path = Path(data_path)
    for f in os.listdir(data_path):
        if f.endswith('.tif') or f.endswith('.tiff'):
            imsave(data_path / f, imread(data_path / f), compress=2)


def decompress_tifffiles(data_path):
    """
    Decompresses tiff files in a folder.
    Args:
        data_path: path to a folder containing tiff files

    """
    data_path = Path(data_path)
    for element in os.listdir(data_path):
        if element.endswith('.tif') or element.endswith('.tiff'):
            imsave(data_path / element, imread(data_path / element))


def collect_leaf_paths(root_paths):
    """Collects all paths to leaf folders."""
    leaf_paths = [p for p in Path(root_paths).glob('**') if not os.walk(p).__next__()[1]]
    return leaf_paths


def compute_seeds(mask):
    """
    Computes a set of seed points
    Args:
        mask: tuple of segmentation mask indices

    Returns: array of seed points

    """
    mask = np.array(mask)

    box_shape = np.max(mask, axis=1) - np.min(mask, axis=1) + 3  # add background border
    dummy = np.zeros(tuple(box_shape))
    dummy[tuple(mask - np.min(mask, axis=1).reshape(-1, 1) + 1)] = 1
    dist = distance_transform_edt(dummy)
    stacked = np.stack(np.gradient(dist))
    abs_grad = np.sum(stacked**2, axis=0)
    seed_points = np.where((abs_grad < 0.1) & (dist > 0))
    if len(seed_points[0]) < 1:
        seed_points = np.median(mask, axis=-1).reshape(-1, 1)
    else:
        # compute non shifted position
        seed_points = np.array(seed_points) + np.min(mask, axis=1).reshape(-1, 1) - 1
        index = np.random.choice(len(seed_points[0]), min(len(seed_points[0]), 100), replace=False)
        seed_points = seed_points[..., index]
    return seed_points
