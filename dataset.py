import logging
import os
from pathlib import Path

import numpy as np
import pycolmap
import torch
from hloc.pipelines.RobotCar.pipeline import CONDITIONS
from skimage import color
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import colmap_read
import benchmark_utils

_logger = logging.getLogger(__name__)


def read_intrinsic(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    name2params = {}
    for line in lines:
        img_name, cam_type, w, h, f, cx, cy, k = line.split(" ")
        f, cx, cy, k = map(float, [f, cx, cy, k])
        w, h = map(int, [w, h])
        name2params[img_name] = [cam_type, w, h, f, cx, cy, k]
    return name2params


def read_train_poses(a_file):
    with open(a_file) as file:
        lines = [line.rstrip() for line in file]
    name2mat = {}
    for line in lines:
        img_name, *matrix = line.split(" ")
        if matrix:
            matrix = np.array(matrix, float).reshape(4, 4)
        name2mat[img_name] = matrix
    return name2mat


class CambridgeLandmarksDataset(Dataset):
    def __init__(self, root_dir, ds_name, train=True):
        self.using_sfm_poses = True
        self.image_name2id = None
        self.train = train
        self.ds_type = ds_name

        # Setup data paths.
        root_dir = Path(root_dir)
        self.sfm_model_dir = root_dir / "reconstruction.nvm"

        if self.train:
            (
                self.xyz_arr,
                self.image2points,
                self.image2name,
                self.image2pose,
                self.image2info,
                self.image2uvs,
                self.rgb_arr,
            ) = utils.read_nvm_file(self.sfm_model_dir)
            self.name2id = {v: u for u, v in self.image2name.items()}

        # pid2images = {}
        # for img in self.image2points:
        #     for pid_ in self.image2points[img]:
        #         pid2images.setdefault(pid_, []).append(img)
        #
        # points = self.image2points[self.name2id["seq4/frame00224.jpg"]]
        # import random
        # pid0 = random.choice(points)
        # images0 = pid2images[pid0]
        # for img in images0:
        #     pose0 = self.image2pose[img]
        #     break
        #
        # import open3d as o3d
        #
        # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz_arr))
        # cl, inlier_ind = point_cloud.remove_radius_outlier(nb_points=16, radius=5)
        # cl.colors = o3d.utility.Vector3dVector(rgb_arr[inlier_ind]/255)
        #
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(width=1920, height=1025)
        # vis.add_geometry(cl)
        # vis.run()
        # vis.destroy_window()

        if self.train:
            root_dir = root_dir / "train"
        else:
            root_dir = root_dir / "test"

        # Main folders.
        rgb_dir = root_dir / "rgb"
        pose_dir = root_dir / "poses"
        calibration_dir = root_dir / "calibration"

        # Find all images. The assumption is that it only contains image files.
        self.rgb_files = sorted(rgb_dir.iterdir())

        # Find all ground truth pose files. One per image.
        self.pose_files = sorted(pose_dir.iterdir())

        # Load camera calibrations. One focal length per image.
        self.calibration_files = sorted(calibration_dir.iterdir())

        self.valid_file_indices = np.arange(len(self.rgb_files))
        self.root_dir = str(root_dir)

    def _load_image(self, idx):
        image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def _load_pose(self, idx):
        pose = np.loadtxt(self.pose_files[idx])
        pose = torch.from_numpy(pose).float()
        return pose

    def _get_single_item(self, idx):
        img_id = self.valid_file_indices[idx]
        image_name = str(self.rgb_files[img_id])

        # Load image.
        image = self._load_image(img_id)

        # Load intrinsics.
        k = np.loadtxt(self.calibration_files[img_id])
        focal_length = float(k)

        # Load pose.
        pose = self._load_pose(img_id)

        # Invert the pose.
        pose_inv = pose.inverse()

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)

        intrinsics[0, 0] = focal_length
        intrinsics[1, 1] = focal_length
        intrinsics[0, 2] = image.shape[1] / 2  # 427
        intrinsics[1, 2] = image.shape[0] / 2  # 240

        if self.train:
            key_ = image_name.split("/")[-1].replace("_", "/").replace("png", "jpg")
            key_ = self.name2id[key_]
            pid_list = self.image2points[key_]
            xyz_gt = self.xyz_arr[pid_list]

            uv_gt = utils.project_using_pose(
                pose_inv.unsqueeze(0).cuda().float(),
                intrinsics.unsqueeze(0).cuda().float(),
                xyz_gt,
            )
        else:
            pid_list = []
            xyz_gt = []
            uv_gt = []

        focal_length = intrinsics[0, 0].item()
        c1 = intrinsics[0, 2].item()
        c2 = intrinsics[1, 2].item()

        camera = {
            "model": "SIMPLE_PINHOLE",
            "height": image.shape[0],
            "width": image.shape[1],
            "params": [focal_length, c1, c2],
        }

        return (
            image,
            image_name,
            img_id,
            pid_list,
            pose_inv,
            None,
            camera,
            xyz_gt,
            uv_gt,
        )

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


class AachenDataset(Dataset):
    def __init__(self, ds_dir="datasets/aachen_v1.1", train=True):
        self.ds_type = "aachen"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/3D-models/aachen_v_1_1"
        self.images_dir_str = f"{self.ds_dir}/images_upright"
        self.images_dir = Path(self.images_dir_str)

        self.train = train
        self.day_intrinsic_file = (
            f"{self.ds_dir}/queries/day_time_queries_with_intrinsics.txt"
        )
        self.night_intrinsic_file = (
            f"{self.ds_dir}/queries/night_time_queries_with_intrinsics.txt"
        )

        if self.train:
            self.recon_images = colmap_read.read_images_binary(
                f"{self.sfm_model_dir}/images.bin"
            )
            self.image_name2id = {}
            for image_id, image in self.recon_images.items():
                self.image_name2id[image.name] = image_id
            self.img_ids = list(self.image_name2id.values())
        else:
            name2params1 = read_intrinsic(self.day_intrinsic_file)
            name2params2 = read_intrinsic(self.night_intrinsic_file)
            self.name2params = {**name2params1, **name2params2}
            self.img_ids = list(self.name2params.keys())
        return

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]
            name = self.recon_images[img_id].name
            image_name = str(self.images_dir / name)
            qvec = self.recon_images[img_id].qvec
            tvec = self.recon_images[img_id].tvec
            # pose_inv = benchmark_utils.return_pose_mat_no_inv(qvec, tvec)
            quat = np.concatenate([qvec, tvec])

        else:
            name1 = self.img_ids[idx]
            image_name = str(self.images_dir / name1)

            img_id = name1
            quat = None

        return image_name, img_id, quat

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


class RobotCarDataset(Dataset):
    def __init__(self, ds_dir="datasets/robotcar", train=True, evaluate=False):
        self.ds_type = "robotcar"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/3D-models/all-merged/all.nvm"
        self.images_dir = Path(f"{self.ds_dir}/images")
        self.test_file1 = f"{ds_dir}/robotcar_v2_train.txt"
        self.test_file2 = f"{ds_dir}/robotcar_v2_test.txt"
        self.ds_dir_path = Path(self.ds_dir)
        self.images_dir_str = str(self.images_dir)
        self.train = train
        self.evaluate = evaluate
        if evaluate:
            assert not self.train

        if self.train:
            (
                self.image2name,
                self.image2pose,
            ) = benchmark_utils.read_nvm_file(self.sfm_model_dir)
            self.name2image = {v: k for k, v in self.image2name.items()}
            self.img_ids = list(self.image2name.keys())

            self.name2mat = read_train_poses(self.test_file1)
        else:
            self.ts2cond = {}
            for condition in CONDITIONS:
                all_image_names = list(Path.glob(self.images_dir, f"{condition}/*/*"))

                for name in all_image_names:
                    time_stamp = str(name).split("/")[-1].split(".")[0]
                    self.ts2cond.setdefault(time_stamp, []).append(condition)
            for ts in self.ts2cond:
                assert len(self.ts2cond[ts]) == 3

            if not self.evaluate:
                self.name2mat = read_train_poses(self.test_file1)
            else:
                self.name2mat = read_train_poses(self.test_file2)
            self.img_ids = list(self.name2mat.keys())

        return

    def _process_id_to_name(self, img_id):
        name = self.image2name[img_id].split("./")[-1]
        name2 = str(self.images_dir / name).replace(".png", ".jpg")
        return name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]
            image_name = self._process_id_to_name(img_id)
            quat = None
            if type(self.image2pose[img_id]) == list:
                qw, qx, qy, qz, tx, ty, tz = self.image2pose[img_id]
                tx, ty, tz = -(Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ np.array([tx, ty, tz]))
                quat = qw, qx, qy, qz, tx, ty, tz
            else:
                pose_mat = self.image2pose[img_id]
                # pose_mat = np.linalg.inv(pose_mat)
        else:
            name0 = self.img_ids[idx]
            try:
                pose_mat = self.name2mat[name0]
                # pose_mat = np.linalg.inv(pose_mat)
                qx, qy, qz, qw = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
                tx, ty, tz = pose_mat[:3, 3]
                quat = qw, qx, qy, qz, tx, ty, tz
            except:
                quat = None

            if self.evaluate:
                time_stamp = str(name0).split("/")[-1].split(".")[0]
                cond = self.ts2cond[time_stamp][0]
                name1 = f"{cond}/{name0}"
                if ".png" in name1:
                    name1 = name1.replace(".png", ".jpg")
            else:
                name1 = name0

            image_name = str(self.images_dir / name1)

            img_id = name1

        return (
            image_name,
            img_id,
            quat,
        )

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


class CMUDataset(Dataset):
    def __init__(self, ds_dir="datasets/datasets/cmu_extended/slice2", train=True):
        self.ds_type = f"cmu/{ds_dir.split('/')[-1]}"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/sparse"
        self.intrinsics_dict = {
            "c0": pycolmap.Camera(
                model="OPENCV",
                width=1024,
                height=768,
                params=[
                    868.993378,
                    866.063001,
                    525.942323,
                    420.042529,
                    -0.399431,
                    0.188924,
                    0.000153,
                    0.000571,
                ],
            ),
            "c1": pycolmap.Camera(
                model="OPENCV",
                width=1024,
                height=768,
                params=[
                    873.382641,
                    876.489513,
                    529.324138,
                    397.272397,
                    -0.397066,
                    0.181925,
                    0.000176,
                    -0.000579,
                ],
            ),
        }
        if train:
            self.images_dir_str = f"{self.ds_dir}/database"
            self.images_dir = Path(self.images_dir_str)
            self.recon_images = colmap_read.read_images_binary(
                f"{self.sfm_model_dir}/images.bin"
            )
            self.recon_cameras = colmap_read.read_cameras_binary(
                f"{self.sfm_model_dir}/cameras.bin"
            )
            self.recon_points = colmap_read.read_points3D_binary(
                f"{self.sfm_model_dir}/points3D.bin"
            )
            self.image_name2id = {}
            for image_id, image in self.recon_images.items():
                self.image_name2id[image.name] = image_id
            self.image_id2points = {}
            self.pid2images = {}

            for img_id in self.recon_images:
                pid_arr = self.recon_images[img_id].point3D_ids
                pid_arr = pid_arr[pid_arr >= 0]
                xyz_arr = np.zeros((pid_arr.shape[0], 3))
                for idx, pid in enumerate(pid_arr):
                    xyz_arr[idx] = self.recon_points[pid].xyz
                self.image_id2points[img_id] = xyz_arr
            self.img_ids = list(self.image_name2id.values())
        else:
            self.images_dir_str = f"{self.ds_dir}/query"
            self.images_dir = Path(self.images_dir_str)

            self.img_ids = [
                str(file) for file in self.images_dir.iterdir() if file.is_file()
            ]

        self.train = train

    def clear(self):
        if self.train:
            self.recon_images.clear()
            self.recon_cameras.clear()
            self.recon_points.clear()

    def _load_image(self, img_id):
        if self.train:
            name = self.recon_images[img_id].name
            name2 = str(self.images_dir / name)
        else:
            name2 = img_id
        try:
            image = io.imread(name2)
        except ValueError:
            return None, name2

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image, name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        img_id = self.img_ids[idx]
        image, image_name = self._load_image(img_id)
        if image is None:
            print(f"Warning: cannot read image at {image_name}")
            return None

        if self.train:
            camera_id = self.recon_images[img_id].camera_id
            camera = self.recon_cameras[camera_id]
            camera = pycolmap.Camera(
                model=camera.model,
                width=int(camera.width),
                height=int(camera.height),
                params=camera.params,
            )
            qvec = self.recon_images[img_id].qvec
            tvec = self.recon_images[img_id].tvec
            pose_inv = utils.return_pose_mat_no_inv(qvec, tvec)

            xyz_gt = self.image_id2points[img_id]
            pid_list = self.recon_images[img_id].point3D_ids
            mask = pid_list >= 0
            pid_list = pid_list[mask]
            uv_gt = self.recon_images[img_id].xys[mask]

            pose_inv = torch.from_numpy(pose_inv)

        else:
            cam_id = image_name.split("/")[-1].split("_")[2]
            camera = self.intrinsics_dict[cam_id]

            image = None
            img_id = image_name.split("/")[-1]
            pid_list = []
            pose_inv = None
            xyz_gt = None
            uv_gt = None

        return (
            image,
            image_name,
            img_id,
            pid_list,
            pose_inv,
            None,
            camera,
            xyz_gt,
            uv_gt,
        )

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)
