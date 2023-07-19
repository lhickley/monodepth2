import os
import numpy as np
import skimage.transform
import PIL.Image as pil

from .mono_dataset import MonoDataset

class CityscapesDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)

        # Intrinsics matrix for Cityscapes (Update these values according to the dataset)
        self.fx = ...  # Focal length in pixels (horizontal)
        self.fy = ...  # Focal length in pixels (vertical)
        self.cx = ...  # Principal point x-coordinate in pixels
        self.cy = ...  # Principal point y-coordinate in pixels

        # Normalize the intrinsics matrix
        self.K = np.array([[self.fx / self.width, 0, self.cx / self.width, 0],
                           [0, self.fy / self.height, self.cy / self.height, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (2048, 1024)  # Adjust according to Cityscapes image size
        self.side_map = {"left": "l", "right": "r"}

    def check_depth(self, folder, frame_index, side):
        """Check if ground truth depth maps are available for the given frame.

        Args:
            folder (str): The folder or sequence name in the Cityscapes dataset.
            frame_index (int): The frame number or identifier.
            side (str): The camera view, like 'left' or 'right'.

        Returns:
            bool: True if the ground truth depth map is available, False otherwise.
        """
        f_str = "{:06d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "disparity",
            folder.split('/')[-1] + "_" + f_str.replace(".png", "_" + self.side_map[side] + ".png"))

        return os.path.isfile(depth_path)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "leftImg8bit",
            folder.split('/')[-1] + "_" + f_str.replace(".png", "_" + self.side_map[side] + ".png"))
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "disparity",
            folder.split('/')[-1] + "_" + f_str.replace(".png", "_" + self.side_map[side] + ".png"))

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
