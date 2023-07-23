# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

#TODO Make this generic
#TODO make the inputs to the depth from disparity read from the correct json files.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map
from cityscapes_utils import generate_depth_map_cityscapes

import numpy as np

def generate_depth_from_disparity(disparity, f, B):
    """
    Convert disparity map to depth map.

    Parameters:
    - disparity: 2D numpy array of disparity values
    - f: focal length of the camera in pixels
    - B: baseline of the stereo camera setup in meters or millimeters

    Returns:
    - depth: 2D numpy array of depth values
    """
    
    # Ensure no division by zero
    disparity = np.clip(disparity, 1e-8, None)

    # Calculate depth
    depth = (f * B) / disparity

    return depth

def export_gt_depths_cityscapes():
    parser = argparse.ArgumentParser(description='export_gt_depth_cityscapes')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the Cityscapes data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["cityscapes", "cityscapes_val"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []

    count=0

    for idx, line in enumerate(lines):

        side, use, folder, first_id, frame_id = line.split()
        frame_id = int(frame_id)

        if opt.split == "cityscapes":
            disparity_path = os.path.join(opt.data_path, folder, "{}_{}_{:06d}_disparity.png".format(folder, first_id, frame_id))
            disparity = np.array(pil.open(disparity_path)).astype(np.float32)
            #gt_depth = generate_depth_map_cityscapes()
            gt_depth = generate_depth_from_disparity(disparity, 2262.52, 0.209313)  # Convert disparity to depth

        if gt_depth.shape == gt_depths[0].shape if gt_depths else True:
            gt_depths.append(gt_depth.astype(np.float32))
            print(line)
        else:
            print("Skipping frame with inhomogeneous shape: {}".format(line))
            count+=1

    output_path = os.path.join(split_folder, "gt_depths.npz")
    print(count)
    print("Saving to {}".format(opt.split))
    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_cityscapes()

'''
def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []

    count=0

    print(len(lines))

    for idx, line in enumerate(lines):

        #print(line)

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        if idx == 0:
            gt_depths.append(gt_depth.astype(np.float32))
            print(line)
        elif gt_depth.shape == gt_depths[0].shape:
            gt_depths.append(gt_depth.astype(np.float32))
            print(line)
        else:
            print("Skipping frame with inhomogeneous shape: {}".format(line))
            count+=1

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print(count)

    print("Saving to {}".format(opt.split))

    #print(gt_depths)

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
'''