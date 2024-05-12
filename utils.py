import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def read_nvm_file(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    nb_cameras = int(lines[2])
    image2info = {}
    image2pose = {}
    image2name = {}
    unique_names = []
    for i in tqdm(range(nb_cameras), desc="Reading cameras"):
        cam_info = lines[3 + i]
        if "\t" in cam_info:
            img_name, info = cam_info.split("\t")
            focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = map(float, info.split(" "))
        else:
            img_name, focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = cam_info.split(" ")
            focal, qw, qx, qy, qz, tx, ty, tz, radial = map(
                float, [focal, qw, qx, qy, qz, tx, ty, tz, radial]
            )
        image2name[i] = img_name
        assert img_name not in unique_names
        unique_names.append(img_name)
        image2info[i] = [focal, radial]
        image2pose[i] = [qw, qx, qy, qz, tx, ty, tz]
    nb_points = int(lines[4 + nb_cameras])
    image2points = {}
    image2uvs = {}
    xyz_arr = np.zeros((nb_points, 3), np.float64)
    rgb_arr = np.zeros((nb_points, 3), np.float64)
    # nb_points = 100
    for j in tqdm(range(nb_points), desc="Reading points"):
        point_info = lines[5 + nb_cameras + j].split(" ")
        x, y, z, r, g, b, nb_features = point_info[:7]
        x, y, z = map(float, [x, y, z])
        xyz_arr[j] = [x, y, z]
        rgb_arr[j] = [r, g, b]
        features_info = point_info[7:]
        nb_features = int(nb_features)
        for k in range(nb_features):
            image_id, _, u, v = features_info[k * 4 : (k + 1) * 4]
            image_id = int(image_id)
            u, v = map(float, [u, v])
            image2points.setdefault(image_id, []).append(j)
            image2uvs.setdefault(image_id, []).append([u, v])

    return xyz_arr, image2points, image2name, image2pose, image2info, image2uvs, rgb_arr


def project_using_pose(gt_pose_inv_B44, intrinsics_B33, xyz):
    xyzt = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyzt = torch.from_numpy(xyzt).permute([1, 0]).float().cuda()

    gt_inv_pose_34 = gt_pose_inv_B44[0, :3]
    cam_coords = torch.mm(gt_inv_pose_34, xyzt)
    uv = torch.mm(intrinsics_B33[0], cam_coords)
    uv[2].clamp_(min=0.1)  # avoid division by zero
    uv = uv[0:2] / uv[2]
    uv = uv.permute([1, 0]).cpu().numpy()
    return uv


def return_pose_mat_no_inv(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    return pose_4x4


def read_kp_and_desc(name, features_h5):
    pred = {}
    img_id = "/".join(name.split("/")[-2:])
    try:
        grp = features_h5[img_id]
    except KeyError:
        grp = features_h5[name]
    for k, v in grp.items():
        pred[k] = v

    pred = {k: np.array(v) for k, v in pred.items()}
    scale = pred["scale"]
    keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
    descriptors = pred["descriptors"].T
    return keypoints, descriptors
