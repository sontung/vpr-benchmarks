import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from types import SimpleNamespace
import PIL
import cv2
import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load


def read_nvm_file(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    nb_cameras = int(lines[2])
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
        image2pose[i] = [qw, qx, qy, qz, tx, ty, tz]

    return image2name, image2pose


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


def hloc_conf_for_all_models():
    conf = {
        "superpoint": {
            "output": "feats-superpoint-n4096-r1024",
            "model": {
                "name": "superpoint",
                "nms_radius": 3,
                "max_keypoints": 4096,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": 1024,
            },
        },
        "r2d2": {
            "output": "feats-r2d2-n5000-r1024",
            "model": {
                "name": "r2d2",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1024,
            },
        },
        "d2net": {
            "output": "feats-d2net-ss",
            "model": {
                "name": "d2net",
                "multiscale": False,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "sift": {
            "output": "feats-sift",
            "model": {"name": "dog"},
            "preprocessing": {
                "grayscale": True,
                "resize_max": 1600,
            },
        },
        "disk": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "netvlad": {
            "output": "global-feats-netvlad",
            "model": {"name": "netvlad"},
            "preprocessing": {"resize_max": 1024},
        },
        "openibl": {
            "output": "global-feats-openibl",
            "model": {"name": "openibl"},
            "preprocessing": {"resize_max": 1024},
        },
        "eigenplaces": {
            "output": "global-feats-eigenplaces",
            "model": {"name": "eigenplaces"},
            "preprocessing": {"resize_max": 1024},
        },
    }

    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }
    return conf, default_conf


def read_global_desc(name, global_features_h5):
    img_id = "/".join(name.split("/")[-2:])
    try:
        desc = np.array(global_features_h5[name]["global_descriptor"])
    except KeyError:
        desc = np.array(global_features_h5[img_id]["global_descriptor"])
    return desc


def write_to_h5_file(fd, name, dict_):
    img_id = "/".join(name.split("/")[-2:])
    name = img_id
    try:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        for k, v in dict_.items():
            grp.create_dataset(k, data=v)
    except OSError as error:
        if "No space left on device" in error.args[0]:
            print("No space left")
            del grp, fd[name]
        raise error


def prepare_encoders(retrieval_model, global_desc_dim):
    conf, default_conf = hloc_conf_for_all_models()

    if retrieval_model == "mixvpr":
        from mix_vpr_model import MVModel

        encoder_global = MVModel(global_desc_dim)
        conf_ns_retrieval = None
    elif retrieval_model == "crica":
        from crica_model import CricaModel

        encoder_global = CricaModel()
        conf_ns_retrieval = None
    elif retrieval_model == "salad":
        from salad_model import SaladModel

        encoder_global = SaladModel()
        conf_ns_retrieval = None
    elif retrieval_model == "gcl":
        from gcl_model import GCLModel

        encoder_global = GCLModel()
        conf_ns_retrieval = None
    else:
        model_dict = conf[retrieval_model]["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        if retrieval_model == "eigenplaces":
            model_dict.update(
                {
                    "variant": "EigenPlaces",
                    "backbone": "ResNet101",
                    "fc_output_dim": global_desc_dim,
                }
            )
            encoder_global = Model(model_dict).eval().to(device)
            encoder_global.conf["name"] = f"eigenplaces_{model_dict['backbone']}"
        else:
            encoder_global = Model(model_dict).eval().to(device)
        conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"][
            "resize_max"
        ]
    return encoder_global, conf_ns_retrieval


def read_and_preprocess(name, conf):
    image = read_image_by_hloc(name, conf.grayscale)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    scale = 1

    if conf.resize_max and (conf.resize_force or max(size) > conf.resize_max):
        scale = conf.resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = resize_image_by_hloc(image, size_new, conf.interpolation)

    if conf.grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0

    return image, scale


def read_image_by_hloc(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image_by_hloc(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized
