import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import faiss
import h5py
import numpy as np
import poselib
import torch
from pykdtree.kdtree import KDTree
from tqdm import tqdm


import benchmark_utils
from dataset import RobotCarDataset


def retrieve_pid(pid_list, uv_gt, keypoints):
    tree = KDTree(keypoints.astype(uv_gt.dtype))
    dis, ind = tree.query(uv_gt)
    mask = dis < 5
    selected_pid = np.array(pid_list)[mask]
    return selected_pid, mask, ind


def compute_pose_error(pose, pose_gt):
    est_pose = np.vstack([pose.Rt, [0, 0, 0, 1]])
    out_pose = torch.from_numpy(est_pose)

    # Calculate translation error.
    t_err = float(torch.norm(pose_gt[0:3, 3] - out_pose[0:3, 3]))

    gt_R = pose_gt[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3].numpy()

    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return t_err, r_err


def combine_descriptors(local_desc, global_desc, lambda_value_):
    res = (
        lambda_value_ * local_desc
        + (1 - lambda_value_) * global_desc[: local_desc.shape[1]]
    )
    return res


class BaseTrainer:
    def __init__(
        self,
        train_ds,
        test_ds,
        global_feature_dim,
        global_desc_model,
        global_desc_conf,
    ):
        self.dataset = train_ds
        self.test_dataset = test_ds
        self.global_feature_dim = global_feature_dim

        self.name2uv = {}
        self.ds_name = self.dataset.ds_type
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.global_desc_model_name = (
            f"{global_desc_model.conf['name']}_{global_feature_dim}"
        )

        self.global_desc_model = global_desc_model
        self.global_desc_conf = global_desc_conf

        self.image2desc = self.collect_image_descriptors()

        self.xyz_arr = None
        self.map_reduction = False
        self.all_desc, self.all_names, self.all_poses = self.collect_image_descriptors()

    def collect_image_descriptors(self):
        file_name1 = f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
        file_name3 = f"output/{self.ds_name}/image_desc_quat_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
        if os.path.isfile(file_name1):
            all_desc = np.load(file_name1)
            all_poses = np.load(file_name3)
            afile = open(file_name2, "rb")
            all_names = pickle.load(afile)
            afile.close()
        else:
            all_desc = np.zeros((len(self.dataset), self.global_feature_dim))
            all_poses = np.zeros((len(self.dataset), 7))
            all_names = []
            idx = 0
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Collecting image descriptors"):
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[0])
                    all_desc[idx] = image_descriptor
                    all_poses[idx] = example[2]
                    all_names.append(example[0])
                    idx += 1
            np.save(file_name1, all_desc)
            np.save(file_name3, all_poses)
            with open(file_name2, "wb") as handle:
                pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return all_desc, all_names, all_poses

    def produce_image_descriptor(self, name):
        with torch.no_grad():
            if (
                "mixvpr" in self.global_desc_model_name
                or "crica" in self.global_desc_model_name
                or "salad" in self.global_desc_model_name
                or "gcl" in self.global_desc_model_name
                or "patchnetvlad" in self.global_desc_model_name
            ):
                image_descriptor = self.global_desc_model.process(name)
            else:
                image, _ = benchmark_utils.read_and_preprocess(
                    name, self.global_desc_conf
                )
                image_descriptor = (
                    self.global_desc_model(
                        {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                    )["global_descriptor"]
                    .squeeze()
                    .cpu()
                    .numpy()
                )
        return image_descriptor

    def evaluate(self, result_file_name, return_results=False):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.global_feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.all_desc)
        result_file = open(result_file_name, "w")
        all_results = []

        global_descriptors_path = f"output/{self.ds_name}/{self.global_desc_model_name}_{self.global_feature_dim}_desc_test.h5"
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Collecting global descriptors for test set"
                ):
                    image_descriptor = self.produce_image_descriptor(example[0])
                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    benchmark_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()
        global_features_h5 = h5py.File(global_descriptors_path, "r")

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                image_descriptor = benchmark_utils.read_global_desc(
                    name, global_features_h5
                )
                _, pose_idx = gpu_index_flat.search(
                    image_descriptor.reshape((1, -1)), 1
                )
                pose_idx = int(pose_idx)

                qw, qx, qy, qz, tx, ty, tz = self.all_poses[pose_idx]
                qvec = " ".join(map(str, [qw, qx, qy, qz]))
                tvec = " ".join(map(str, [tx, ty, tz]))

                image_id = self.process_image_id(example)
                line = f"{image_id} {qvec} {tvec}"
                if return_results:
                    all_results.append(line)
                print(line, file=result_file)
        result_file.close()
        global_features_h5.close()
        return all_results

    def process_image_id(self, example):
        image_id = example[1].split("/")[-1]
        return image_id


class RobotCarTrainer(BaseTrainer):
    def process_image_id(self, example):
        image_id = "/".join(example[1].split("/")[1:])
        return image_id


class CambridgeLandmarksTrainer(BaseTrainer):
    def collect_descriptors(self, vis=False):
        features_h5 = self.load_local_features()
        pid2descriptors = {}
        for example in tqdm(self.dataset, desc="Collecting point descriptors"):
            keypoints, descriptors = dd_utils.read_kp_and_desc(example[1], features_h5)
            pid_list = example[3]
            uv = example[-1]
            selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
            idx_arr, ind2 = np.unique(ind[mask], return_index=True)

            selected_descriptors = descriptors[idx_arr]
            if self.using_global_descriptors:
                image_descriptor = self.image2desc[example[1]]
                selected_descriptors = combine_descriptors(
                    selected_descriptors, image_descriptor, self.lambda_val
                )

            for idx, pid in enumerate(selected_pid[ind2]):
                if pid not in pid2descriptors:
                    pid2descriptors[pid] = selected_descriptors[idx]
                else:
                    pid2descriptors[pid] = 0.5 * (
                        pid2descriptors[pid] + selected_descriptors[idx]
                    )

        features_h5.close()
        self.image2desc.clear()

        all_pid = list(pid2descriptors.keys())
        all_pid = np.array(all_pid)
        pid2mean_desc = np.zeros(
            (all_pid.shape[0], self.feature_dim),
            pid2descriptors[list(pid2descriptors.keys())[0]].dtype,
        )

        for ind, pid in enumerate(all_pid):
            pid2mean_desc[ind] = pid2descriptors[pid]

        if pid2mean_desc.shape[0] > all_pid.shape[0]:
            pid2mean_desc = pid2mean_desc[all_pid]
        self.xyz_arr = self.dataset.xyz_arr[all_pid]
        return pid2mean_desc, all_pid, {}

    def legal_predict(
        self,
        uv_arr,
        features_ori,
        gpu_index_flat,
        remove_duplicate=False,
        return_pid=False,
    ):
        distances, feature_indices = gpu_index_flat.search(features_ori, 1)

        feature_indices = feature_indices.ravel()

        if remove_duplicate:
            pid2uv = {}
            for idx in range(feature_indices.shape[0]):
                pid = feature_indices[idx]
                dis = distances[idx][0]
                uv = uv_arr[idx]
                if pid not in pid2uv:
                    pid2uv[pid] = [dis, uv]
                else:
                    if dis < pid2uv[pid][0]:
                        pid2uv[pid] = [dis, uv]
            uv_arr = np.array([pid2uv[pid][1] for pid in pid2uv])
            feature_indices = [pid for pid in pid2uv]

        pred_scene_coords_b3 = self.xyz_arr[feature_indices]
        if return_pid:
            return uv_arr, pred_scene_coords_b3, feature_indices

        return uv_arr, pred_scene_coords_b3

    def evaluate(self, return_name2err=False):
        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc)

        global_descriptors_path = (
            f"output/{self.ds_name}/{self.global_desc_model_name}_desc_test.h5"
        )
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Collecting global descriptors for test set"
                ):
                    image_descriptor = self.produce_image_descriptor(example[1])
                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    dd_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()

        features_h5 = h5py.File(self.test_features_path, "r")
        global_features_h5 = h5py.File(global_descriptors_path, "r")
        rErrs = []
        tErrs = []
        testset = self.test_dataset
        name2err = {}
        with torch.no_grad():
            for example in tqdm(testset, desc="Computing pose for test set"):
                name = "/".join(example[1].split("/")[-2:])
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)
                if self.using_global_descriptors:
                    image_descriptor = dd_utils.read_global_desc(
                        name, global_features_h5
                    )

                    descriptors = combine_descriptors(
                        descriptors, image_descriptor, self.lambda_val
                    )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                camera = example[6]
                pose, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera,
                )

                t_err, r_err = compute_pose_error(pose, example[4])
                name2err[name] = t_err

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

        features_h5.close()
        global_features_h5.close()
        total_frames = len(rErrs)
        assert total_frames == len(testset)

        # Compute median errors.
        tErrs.sort()
        rErrs.sort()
        median_idx = total_frames // 2
        median_rErr = rErrs[median_idx]
        median_tErr = tErrs[median_idx]
        if return_name2err:
            return median_tErr, median_rErr, name2err
        return median_tErr, median_rErr
