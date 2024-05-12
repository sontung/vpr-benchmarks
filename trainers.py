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

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.global_feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.all_desc)
        result_file = open(
            f"output/{self.ds_name}/Aachen_v1_1_eval_{self.global_desc_model_name}_{self.global_feature_dim}.txt",
            "w",
        )

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
                    name = example[0]
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

                image_id = example[1].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        result_file.close()
        global_features_h5.close()


class RobotCarTrainer(BaseTrainer):
    def reduce_map_size(self):
        if self.map_reduction:
            return
        index_map_file_name = f"output/{self.ds_name}/indices.npy"
        if os.path.isfile(index_map_file_name):
            inlier_ind = np.load(index_map_file_name)
        else:
            import open3d as o3d

            point_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.dataset.xyz_arr)
            )
            cl, inlier_ind = point_cloud.remove_radius_outlier(
                nb_points=16, radius=5, print_progress=True
            )

            np.save(index_map_file_name, np.array(inlier_ind))

            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1920, height=1025)
            # vis.add_geometry(point_cloud)
            # vis.run()
            # vis.destroy_window()
        img2points2 = {}
        inlier_ind_set = set(inlier_ind)
        for img in tqdm(
            self.dataset.image2points, desc="Removing outlier points in the map"
        ):
            pid_list = self.dataset.image2points[img]
            img2points2[img] = [pid for pid in pid_list if pid in inlier_ind_set]
            mask = [True if pid in inlier_ind_set else False for pid in pid_list]
            self.dataset.image2uvs[img] = np.array(self.dataset.image2uvs[img])[mask]
        self.dataset.image2points = img2points2
        self.map_reduction = True

    def index_db_points(self):
        self.reduce_map_size()
        file_name_for_saving = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_db_info.pkl"
        )
        file_name_for_saving2 = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_db_selected_desc.pkl"
        )
        if os.path.isfile(file_name_for_saving) and os.path.isfile(
            file_name_for_saving2
        ):
            afile = open(file_name_for_saving, "rb")
            self.image2info3d = pickle.load(afile)
            afile.close()
            afile = open(file_name_for_saving2, "rb")
            self.image2selected_desc = pickle.load(afile)
            afile.close()
        else:
            features_h5 = self.load_local_features()
            # features_h5 = h5py.File(
            #     "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/d2net_features_train.h5",
            #     "r")
            self.image2info3d = {}
            self.image2selected_desc = {}
            for example in tqdm(self.dataset, desc="Indexing database points"):
                keypoints, descriptors = dd_utils.read_kp_and_desc(
                    example[1], features_h5
                )
                pid_list = example[3]
                uv = example[-1]
                selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)
                self.image2info3d[example[1]] = [selected_pid, mask, ind, idx_arr, ind2]
                selected_descriptors = descriptors[idx_arr]
                self.image2selected_desc[example[1]] = selected_descriptors

            features_h5.close()
            with open(file_name_for_saving, "wb") as handle:
                pickle.dump(self.image2info3d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(file_name_for_saving2, "wb") as handle:
                pickle.dump(
                    self.image2selected_desc, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

    def improve_codebook(self, vis=False):
        self.reduce_map_size()
        img_dir_str = self.dataset.images_dir_str
        matches_h5 = h5py.File(
            str(f"outputs/robotcar/{self.local_desc_model_name}_nn.h5"),
            # "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/d2net_nn.h5",
            "r",
            libver="latest",
        )
        features_h5 = h5py.File(
            str(f"outputs/robotcar/{self.local_desc_model_name}.h5"),
            # "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/d2net.h5",
            "r",
            libver="latest",
        )

        extra_ds = RobotCarDataset(
            ds_dir=self.dataset.ds_dir, train=False, evaluate=False
        )

        image2desc = {}
        if self.using_global_descriptors:
            for example in tqdm(
                extra_ds, desc="Processing global descriptors for extra images"
            ):
                image_name = example[1]
                image_name_wo_dir = image_name.split(img_dir_str)[-1][1:]
                image_name_for_matching_db = image_name_wo_dir.replace("/", "-")
                if image_name_for_matching_db in matches_h5:
                    image_descriptor = self.produce_image_descriptor(image_name)
                    image2desc[image_name] = image_descriptor

        print(f"Got {len(image2desc)} extra images")
        count = 0
        for example in tqdm(extra_ds, desc="Improving codebook with extra images"):
            image_name = example[1]
            image_name_wo_dir = image_name.split(img_dir_str)[-1][1:]
            image_name_for_matching_db = image_name_wo_dir.replace("/", "-")
            data = matches_h5[image_name_for_matching_db]

            matches_2d_3d = []
            for db_img in data:
                matches = data[db_img]
                indices = np.array(matches["matches0"])
                mask0 = indices > -1
                if np.sum(mask0) < 10:
                    continue
                if len(db_img.split("-")) == 3:
                    db_img_normal = db_img.replace("-", "/")
                else:
                    db_img_normal = db_img.replace("-", "/").replace("/", "-", 1)

                selected_pid, mask, ind, idx_arr, ind2 = self.image2info3d[
                    f"{img_dir_str}/{db_img_normal}"
                ]
                indices = indices[mask0]
                mask2 = np.isin(idx_arr, indices)
                mask3 = np.isin(indices, idx_arr)
                ind2 = ind2[mask2]
                selected_pid = selected_pid[ind2]
                matches_2d_3d.append([mask0, mask3, selected_pid])

            uv0 = np.array(features_h5[image_name_wo_dir]["keypoints"])
            index_arr_for_kp = np.arange(uv0.shape[0])
            all_matches = [[], [], []]
            for mask0, mask3, pid_list in matches_2d_3d:
                uv0_selected = uv0[mask0][mask3]
                indices = index_arr_for_kp[mask0][mask3]
                all_matches[0].append(uv0_selected)
                all_matches[1].extend(pid_list)
                all_matches[2].extend(indices)

            if len(all_matches[1]) < 10:
                tqdm.write(
                    f"Skipping {image_name} because of {len(all_matches[1])} matches"
                )
                continue
            else:
                uv_arr = np.vstack(all_matches[0])
                xyz_pred = self.dataset.xyz_arr[all_matches[1]]
                camera = example[6]

                # camera_dict = {
                #     "model": camera.model.name,
                #     "height": camera.height,
                #     "width": camera.width,
                #     "params": camera.params,
                # }
                # pose, info = poselib.estimate_absolute_pose(
                #     uv_arr,
                #     xyz_pred,
                #     camera_dict,
                # )
                # mask = info["inliers"]

                intrinsics = torch.eye(3)
                focal, cx, cy, _ = camera.params
                intrinsics[0, 0] = focal
                intrinsics[1, 1] = focal
                intrinsics[0, 2] = cx
                intrinsics[1, 2] = cy
                pose_mat = example[4]
                uv_gt = project_using_pose(
                    pose_mat.inverse().unsqueeze(0).cuda().float(),
                    intrinsics.unsqueeze(0).cuda().float(),
                    xyz_pred,
                )
                diff = np.mean(np.abs(uv_gt - uv_arr), 1)
                mask = diff < 5

                count += 1
                descriptors0 = np.array(features_h5[image_name_wo_dir]["descriptors"]).T
                kp_indices = np.array(all_matches[2])[mask]
                pid_list = np.array(all_matches[1])[mask]
                selected_descriptors = descriptors0[kp_indices]
                if self.using_global_descriptors:
                    image_descriptor = image2desc[image_name]
                    selected_descriptors = combine_descriptors(
                        selected_descriptors, image_descriptor, self.lambda_val
                    )

                for idx, pid in enumerate(pid_list):
                    if pid not in self.pid2descriptors:
                        self.pid2descriptors[pid] = selected_descriptors[idx]
                        self.pid2count[pid] = 1
                    else:
                        self.pid2count[pid] += 1
                        self.pid2descriptors[pid] = (
                            self.pid2descriptors[pid] + selected_descriptors[idx]
                        )

        matches_h5.close()
        features_h5.close()
        image2desc.clear()
        print(f"Codebook improved from {count} pairs.")

    def collect_descriptors(self, vis=False):
        self.reduce_map_size()
        features_h5 = self.load_local_features()

        # features_h5 = h5py.File(
        #     "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/d2net_features_train.h5",
        #     "r")

        image2data = {}
        all_pids = []
        image_names = [
            self.dataset._process_id_to_name(img_id) for img_id in self.dataset.img_ids
        ]
        for name in tqdm(image_names, desc="Reading database images"):
            selected_pid, mask, ind, idx_arr, ind2 = self.image2info3d[name]
            if name in self.image2selected_desc:
                selected_descriptors = self.image2selected_desc[name]
            else:
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)
                selected_descriptors = descriptors[idx_arr]
            image2data[name] = [ind2, selected_pid, selected_descriptors]
            all_pids.extend(selected_pid[ind2])

        all_pids = list(set(all_pids))
        all_pids = np.array(all_pids)

        sample0 = list(image2data.keys())[0]
        sample_desc = image2data[sample0][-1]
        if self.using_global_descriptors:
            sample1 = list(self.image2desc.keys())[0]
            sample_desc += self.image2desc[sample1]

        pid2mean_desc = np.zeros(
            (all_pids.shape[0], self.feature_dim), sample_desc.dtype
        )
        pid2count = np.zeros(all_pids.shape[0])
        pid2ind = {pid: idx for idx, pid in enumerate(all_pids)}

        for image_name in tqdm(image_names, desc="Collecting point descriptors"):
            ind2, selected_pid, selected_descriptors = image2data[image_name]
            if self.using_global_descriptors:
                image_descriptor = self.image2desc[image_name]
                selected_descriptors = combine_descriptors(
                    selected_descriptors, image_descriptor, self.lambda_val
                )
            selected_indices = [pid2ind[pid] for pid in selected_pid[ind2]]
            pid2mean_desc[selected_indices] += selected_descriptors
            pid2count[selected_indices] += 1

        for pid in tqdm(self.pid2descriptors, desc="Tuning codebook from extra images"):
            ind = pid2ind[pid]
            pid2mean_desc[ind] += self.pid2descriptors[pid]
            pid2count[ind] += self.pid2count[pid]

        pid2mean_desc = pid2mean_desc / pid2count.reshape(-1, 1)
        features_h5.close()
        self.image2desc.clear()
        self.pid2descriptors.clear()
        self.xyz_arr = self.dataset.xyz_arr[all_pids]
        np.save(
            f"output/{self.ds_name}/pid2mean_desc{self.local_desc_model_name}-{self.global_desc_model_name}-{self.lambda_val}.npy",
            pid2mean_desc,
        )
        np.save(
            f"output/{self.ds_name}/xyz_arr{self.local_desc_model_name}-{self.global_desc_model_name}-{self.lambda_val}.npy",
            self.xyz_arr,
        )
        return pid2mean_desc, all_pids, {}

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

    def evaluate(self):
        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc)

        global_descriptors_path = f"output/{self.ds_name}/{self.global_desc_model_name}_{self.global_feature_dim}_desc_test.h5"
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

        if self.using_global_descriptors:
            result_file = open(
                f"output/{self.ds_name}/RobotCar_eval_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}.txt",
                "w",
            )
        else:
            result_file = open(
                f"output/{self.ds_name}/RobotCar_eval_{self.local_desc_model_name}.txt",
                "w",
            )

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
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
                camera_dict = {
                    "model": camera.model.name,
                    "height": camera.height,
                    "width": camera.width,
                    "params": camera.params,
                }
                pose, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera_dict,
                )

                qvec = " ".join(map(str, pose.q))
                tvec = " ".join(map(str, pose.t))

                image_id = "/".join(example[2].split("/")[1:])
                print(f"{image_id} {qvec} {tvec}", file=result_file)
            result_file.close()
        features_h5.close()
        global_features_h5.close()


class CMUTrainer(BaseTrainer):
    def clear(self):
        del self.pid2mean_desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])

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
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[1])
                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    dd_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()

        features_h5 = h5py.File(self.test_features_path, "r")
        global_features_h5 = h5py.File(global_descriptors_path, "r")
        query_results = []
        print(f"Reading global descriptors from {global_descriptors_path}")
        print(f"Reading local descriptors from {self.test_features_path}")

        if self.using_global_descriptors:
            result_file_name = f"output/{self.ds_name}/CMU_eval_{self.local_desc_model_name}_{self.global_desc_model_name}.txt"
        else:
            result_file_name = (
                f"output/{self.ds_name}/CMU_eval_{self.local_desc_model_name}.txt"
            )

        computed_images = {}
        if os.path.isfile(result_file_name):
            with open(result_file_name) as file:
                lines = [line.rstrip() for line in file]
            if len(lines) == len(self.test_dataset):
                print(f"Found result file at {result_file_name}. Skipping")
                return lines
            else:
                computed_images = {line.split(" ")[0]: line for line in lines}

        result_file = open(result_file_name, "w")
        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                if example is None:
                    continue
                name = example[1]
                image_id = example[2].split("/")[-1]
                if image_id in computed_images:
                    line = computed_images[image_id]
                else:
                    keypoints, descriptors = dd_utils.read_kp_and_desc(
                        name, features_h5
                    )

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

                    camera_dict = {
                        "model": "OPENCV",
                        "height": camera.height,
                        "width": camera.width,
                        "params": camera.params,
                    }
                    pose, info = poselib.estimate_absolute_pose(
                        uv_arr,
                        xyz_pred,
                        camera_dict,
                    )

                    qvec = " ".join(map(str, pose.q))
                    tvec = " ".join(map(str, pose.t))
                    line = f"{image_id} {qvec} {tvec}"
                query_results.append(line)
                print(line, file=result_file)
        features_h5.close()
        global_features_h5.close()
        result_file.close()
        return query_results


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
