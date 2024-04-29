# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Shengjie Xu
# Date: 4/22/2024
# Based on: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nerfstudio_dataparser.py

""" Data parser for nerfstudio datasets. But we add some stuff to load event data as nerfstudio datasets format. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class EventNerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction."""


@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: EventNerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
        # for both fisheye and non-fisheye datasets.
        metadata = {}
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        applied_transform = None
        colmap_path = self.config.data / "colmap/sparse/0"
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        elif colmap_path.exists():
            # For converting from colmap, this was the effective value of applied_transform that was being
            # used before we added the applied_transform field to the output dataformat.
            meta["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)

        if applied_transform is not None:
            dataparser_transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        else:
            dataparser_transform_matrix = transform_matrix

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # reinitialize metadata for dataparser_outputs
        metadata = {}

        # _generate_dataparser_outputs might be called more than once so we check if we already loaded the point cloud
        try:
            self.prompted_user
        except AttributeError:
            self.prompted_user = False

        # Load 3D points
        if self.config.load_3D_points:
            if "ply_file_path" in meta:
                ply_file_path = data_dir / meta["ply_file_path"]

            elif colmap_path.exists():
                from rich.prompt import Confirm

                # check if user wants to make a point cloud from colmap points
                if not self.prompted_user:
                    self.create_pc = Confirm.ask(
                        "load_3D_points is true, but the dataset was processed with an outdated ns-process-data that didn't convert colmap points to .ply! Update the colmap dataset automatically?"
                    )

                if self.create_pc:
                    import json

                    from nerfstudio.process_data.colmap_utils import create_ply_from_colmap

                    with open(self.config.data / "transforms.json") as f:
                        transforms = json.load(f)

                    # Update dataset if missing the applied_transform field.
                    if "applied_transform" not in transforms:
                        transforms["applied_transform"] = meta["applied_transform"]

                    ply_filename = "sparse_pc.ply"
                    create_ply_from_colmap(
                        filename=ply_filename,
                        recon_dir=colmap_path,
                        output_dir=self.config.data,
                        applied_transform=applied_transform,
                    )
                    ply_file_path = data_dir / ply_filename
                    transforms["ply_file_path"] = ply_filename

                    # This was the applied_transform value

                    with open(self.config.data / "transforms.json", "w", encoding="utf-8") as f:
                        json.dump(transforms, f, indent=4)
                else:
                    ply_file_path = None
            else:
                if not self.prompted_user:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                ply_file_path = None

            if ply_file_path:
                sparse_points = self._load_3D_points(ply_file_path, transform_matrix, scale_factor)
                if sparse_points is not None:
                    metadata.update(sparse_points)
            self.prompted_user = True

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath



import os
import numpy as np
from matplotlib import pyplot as plt
import math
import json
import cv2
import imageio
import bm4d
import os
import colour
from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

num_events = 53644

class EventImageDatamanager:
    """
    A datamanager that outputs full images and cameras instead of raybundles.
    This makes the datamanager more lightweight since we don't have to do generate rays.
    Useful for full-image training e.g. rasterization pipelines.
    deblur_method = ["bilinear", "Malvar2004", "Menon2007"]
    The BM4D is the deblur stage, which take quite a long time. Default is off, but it can significantly
    improve synthetic datasets.
    """
    def __init__(self, event_file_path, pose_directory, out_directory, width, height, debayer_method=None, sigma=0):
        self.img_size = (height, width)
        self.debayer = False
        self.is_colored = True
        # self.img = np.zeros(self.img_size, dtype=np.int8)
        self.event_file_path = event_file_path
        self.pose_directory = pose_directory
        self.out_directory = out_directory
        self.F = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]])
        self.F_tile = np.tile(self.F, (int(height/2), int(width/2), 1))
        self.debayer_method = debayer_method
        self.deblur = False

        if sigma>0:
            self.deblur = True
            self.sigma = sigma
        self.getEventData()

        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)

    def loadEventNPZData(self):
        """Idx: Store the event line of the timestep for each output frames. range from [0-999]
           Usage: idx[t_i-1] to idx[t_i] capture events motion image.
                   0         to idx[t_i] capture RGB image
        """
        self.event_data = np.load(self.event_file_path)
        self.timestamp, self.x, self.y, self.pol = self.event_data['t'], self.event_data['x'], self.event_data['y'], self.event_data['p']
        print("Data length:", len(self.timestamp))
        self.idx = []
        j=0
        t = self.timestamp[0]
        frame = 0
        for j in range(107500):
            if self.timestamp[j]>t:
                print("Data shutter:", j)
                break
        start_pt = 0
        for k in range(start_pt,len(self.timestamp)):
            if self.timestamp[k]>t:
                # print('Data streams per frame:',k-start_pt,t)
                start_pt = k
                t+= (self.timestamp[-1] - self.timestamp[0])/1000
                self.idx.append(k-1)
                continue
        print("idx: ", self.idx)
        print("Frame: ", len(self.idx))

    def loadEventTXTData(self):
        infile = open(self.event_file_path, 'r')
        timestamp, x, y, pol = [], [], [], []
        for line in infile:
            words = line.split()
            timestamp.append(float(words[0]))
            x.append(int(words[1]))
            y.append(int(words[2]))
            pol.append(int(words[3]))
        infile.close()
        self.timestamp, self.x, self.y, self.pol = timestamp, x, y, pol

        print("Data length:", len(self.timestamp))
        self.idx = []
        j=0
        t = self.timestamp[0]
        frame = 0
        for j in range(107500):
            if self.timestamp[j]>t:
                print("Data shutter:", j)
                break
        start_pt = 0
        for k in range(start_pt,len(self.timestamp)):
            if self.timestamp[k]>t:
                # print('Data streams per frame:',k-start_pt,t)
                start_pt = k
                t+= (self.timestamp[-1] - self.timestamp[0])/1000
                self.idx.append(k-1)
                continue
        print("idx: ", self.idx)
        print("Frame: ", len(self.idx))

    def getFileType(self):
        root, ext = os.path.splitext(self.event_file_path)
        return ext

    def getEventData(self):
        if self.getFileType() == ".txt":
            print("Loaded txt")
            self.loadEventTXTData()

        if self.getFileType() == ".npz":
            print("Loaded npz")
            self.loadEventNPZData()

    def scale_img(self, array):
        min_val = np.min(array)
        max_val = np.max(array)
        sf = 255 / (max_val - min_val)
        scaled_array = ((array - min_val) * sf).astype(np.uint8)
        return scaled_array

    def convertCameraImg(self, num_events, start=0):
        # print("before:", np.max(img), np.min(img))
        # self.img = np.zeros(self.img_size , dtype=np.float32)
        # self.img = np.zeros(self.img_size , dtype=np.float32) + np.log(125) / 2.2
        self.img = np.zeros(self.img_size , dtype=np.float32) + np.log(127) / 2.2
        self.bayer = np.zeros(self.img_size, np.float32)
        start = start

        # print(self.img[:5,:5])

        print("Load event img at :", num_events)
        self.t_ref = self.timestamp[0] # time of the last event in the packet
        self.tau = 0.03 # decay parameter (in seconds)
        self.dt = num_events*10

        for i in range(start, num_events):
            self.img[self.y[i], self.x[i]] += self.pol[i]

        bg_mask = self.img == np.log(127) / 2.2

        self.img = np.tile(self.img[..., None], (1, 1, 3))
        # img = np.tile(img[..., None], (1, 1, 3)) + np.log(159) / 2.2

        print(self.img[:2,:2])
        print("1. Before:", np.max(self.img), np.min(self.img), self.img.shape)

        self.bayer = self.scale_img(self.img)

        # Apply mask
        # self.bayer[bg_mask] = 198

        self.img_gray = self.bayer
        self.bayer = self.F_tile * self.img_gray
        print("bayer input", np.max(self.bayer), np.min(self.bayer))
        self.bayer = np.clip(np.exp(self.bayer * 2.2), 0, 255).astype(np.uint8)

        if self.debayer_method:
            # mosaic
            self.CFA = mosaicing_CFA_Bayer(self.img_gray)
            print('here')
            # Menon2007
            if self.debayer_method == "bilinear":
                self.bayer = demosaicing_CFA_Bayer_bilinear(self.CFA)
            if self.debayer_method == "Malvar2004":
                self.bayer = demosaicing_CFA_Bayer_Malvar2004(self.CFA)
            if self.debayer_method == "Menon2007":
                self.bayer = demosaicing_CFA_Bayer_Menon2007(self.CFA)
            self.bayer = self.scale_img(self.bayer)
            print("debayer_method", np.max(self.bayer), np.min(self.bayer))

        if self.deblur:
            self.bayer = bm4d.bm4d(self.bayer, self.sigma); # white noise: include noise std
            self.bayer = self.scale_img(self.bayer)

        # # Set background color for RGB
        # self.bayer[bg_mask] = 125

        # # Apply mask
        # self.bayer[bg_mask] = 198
        # self.bayer = np.clip(np.exp(self.bayer * 2.2), 0, 255).astype(np.uint8)
        # self.bayer = np.exp(self.bayer * 2.2)

        print(self.bayer[:2,:2])
        print("2. After:", np.max(self.bayer), np.min(self.bayer), self.bayer.shape)

        return self.img_gray, self.bayer

    def AccuDiffCameraImg(self, t_0, t):
        """ Eq.3 the observed events {Ei}_{i=1}^N between rendered views (multiplied by Bayer colour filter)
        taken at two different time instants t0 and t. (t - t_0).
        """
        return self.convertCameraImg(t, t_0)

    def ImgPlot(self, img):
        fig = plt.figure(figsize=(21,6))
        plt.subplot(1,4,1)
        plt.imshow(img[:,:,0], clim=(0, 255))
        plt.subplot(1,4,2)
        plt.imshow(img[:,:,1], clim=(0, 255))
        plt.subplot(1,4,3)
        plt.imshow(img[:,:,2], clim=(0, 255))
        plt.subplot(1,4,4)
        plt.imshow(img, clim=(0, 255))
        plt.show()

    def ImgSave(self, img, file_path, file_name):
        # Save the image to a PNG file
        imageio.imwrite(file_path + str(file_name) +'.png', img)

    def PoseRead(self, file_path, frame_num):
        file_path = os.path.join(file_path, 'r_'+'{:05d}'.format(frame_num) + ".txt")
        try:
            with open(file_path, 'r') as file:
                # Read each line in the file
                lines = file.readlines()
                # Parse each line to extract the elements of the camera matrix
                camera_matrix = []
                for line in lines:
                    elements = line.split()
                    camera_matrix.append([float(element) for element in elements])
                camera_matrix = np.array(camera_matrix)
                print("test:", camera_matrix)
                return camera_matrix

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def convert_to_json (self):
        AABB_SCALE = 1
        text = os.path.normpath(self.pose_directory) # + '/text')
        OUT_PATH = os.path.normpath(self.out_directory + '/' + "transforms.json")
        # sparce = os.path.normpath(args.scenedir + '/sparse')
                
        # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
        # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
        # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443

        w = self.img_size[1]
        h = self.img_size[0]
        fl_x = 0
        fl_y = 0
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        cx = w / 2
        cy = h / 2

        # angle_x = math.atan(w / (fl_x * 2)) * 2
        # angle_y = math.atan(h / (fl_y * 2)) * 2
        # fovx = angle_x * 180 / math.pi
        # fovy = angle_y * 180 / math.pi

        # with open(os.path.join(text,"images.txt"), "r") as f:
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            # "camera_angle_x": angle_x,
            # "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)

        for i in range(6):
            if  i % 2 == 1:
                # elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                pose_path = str(f"/content/")
                name = str(f"./images/frame_{'{:05d}'.format(i)}.png")
                print(name)
                # b=sharpness(os.path.normpath(f"{args.scenedir}/{args.images}/{elems[9]}"))
                # print(name, "sharpness=",b)
                # image_id = int(elems[0])
                # qvec = np.array(tuple(map(float, elems[1:5])))
                # tvec = np.array(tuple(map(float, elems[5:8])))
                # R = qvec2rotmat(-qvec)
                # t = tvec.reshape([3,1])
                # m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                # c2w = np.linalg.inv(m)
                # c2w[0:3,2] *= -1 # flip the y and z axis
                # c2w[0:3,1] *= -1
                # c2w = c2w[[1,0,2,3],:] # swap y and z
                # c2w[2,:] *= -1 # flip whole world upside down

                # up += c2w[0:3,1]
                
                c2w = self.PoseRead(pose_path, i)
                # frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
                frame={"file_path":name, "transform_matrix": c2w}

                out["frames"].append(frame)

        nframes = len(out["frames"])
        # up = up / np.linalg.norm(up)
        # print("up vector was", up)
        # R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        # R = np.pad(R,[0,1])
        # R[-1, -1] = 1


        # for f in out["frames"]:
        #     f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # # find a central point they are all looking at
        # print("computing center of attention...")
        # totw = 0.0
        # totp = np.array([0.0, 0.0, 0.0])
        # for f in out["frames"]:
        #     mf = f["transform_matrix"][0:3,:]
        #     for g in out["frames"]:
        #         mg = g["transform_matrix"][0:3,:]
        #         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
        #         if w > 0.01:
        #             totp += p*w
        #             totw += w
        # totp /= totw
        # print(totp) # the cameras are looking at totp
        # for f in out["frames"]:
        #     f["transform_matrix"][0:3,3] -= totp

        # avglen = 0.
        # for f in out["frames"]:
        #     avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        # avglen /= nframes
        # print("avg camera distance from origin", avglen)
        # for f in out["frames"]:
        #     f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()

        print(nframes,"frames")
        print(f"writing {OUT_PATH}")
        with open(OUT_PATH, "w") as outfile:
            json.dump(out, outfile, indent=2)


# ## Example
# eventData = EventImageDatamanager(file_path, 346, 260, debayer_method="Menon2007", sigma=0)

# t_0 = eventData.idx[0]
# t = eventData.idx[500]

# print(t_0, t)

# img_t0_gray, img_t0 = eventData.convertCameraImg(t_0)
# img_t_gray, img_t = eventData.convertCameraImg(t)
# img_dt_gray, img_dt = eventData.AccuDiffCameraImg(t_0, t)

# eventData.EventImgPlot(img_t0_gray)
# eventData.EventImgPlot(img_t_gray)
# eventData.EventImgPlot(img_dt_gray)

# eventData.EventImgPlot(img_t0)
# eventData.EventImgPlot(img_t)
# eventData.EventImgPlot(img_dt)

# eventData.ImgSave(img_dt, "/content/", "img_dt")
# eventData.PoseRead("/content/", 5)
# eventData.convert_to_json()