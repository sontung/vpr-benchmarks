import argparse

import numpy as np
from scipy.spatial.transform import Rotation

import benchmark_utils
from dataset import RobotCarDataset
from trainers import RobotCarTrainer


def patchnetvlad_process(train_ds_, test_ds_):
    patchnet_res = "/home/n11373598/hpc-home/work/Patch-NetVLAD/results/PatchNetVLAD_predictions.txt"
    img_dir = "/work/qvpr/data/raw/2020VisualLocalization/RobotCar-Seasons/images/"
    result_file = open("output/robotcar/robotcar_patchnetvlad.txt", "w")
    with open(patchnet_res, 'r') as file:
        lines = [line.rstrip() for line in file][2:]
    image2pose_bundler = benchmark_utils.read_bundle_file("../descriptor-disambiguation/datasets/robotcar/3D-models/all-merged/all.out", "../descriptor-disambiguation/datasets/robotcar/3D-models/all-merged/all.list.txt")

    data = {}
    for line in lines:
        name1, name2 = map(lambda du: du.split(img_dir)[-1], line.split(","))
        if name1 not in data:
            data[name1] = name2

    for example in test_ds_:
        name = example[1]
        name2 = data[name]
        image_id = "/".join(name.split("/")[1:])
        name2_processed = f"./{name2.replace('jpg', 'png')}"
        db_id = train_ds_.name2image[name2_processed]
        pose = train_ds_.image2pose[db_id]
        qw, qx, qy, qz, tx, ty, tz = image2pose_bundler[name2_processed]

        # qw, qx, qy, qz, tx, ty, tz = pose
        # tx, ty, tz = -(Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ np.array([tx, ty, tz]))

        qvec = " ".join(map(str, [qw, qx, qy, qz]))
        tvec = " ".join(map(str, [tx, ty, tz]))

        print(f"{image_id} {qvec} {tvec}", file=result_file)
    result_file.close()


def run_function(
    ds_dir,
    retrieval_model,
    global_desc_dim,
):
    print(f"Using {retrieval_model}-{global_desc_dim}")

    encoder_global, conf_ns_retrieval = benchmark_utils.prepare_encoders(
        retrieval_model, global_desc_dim
    )
    train_ds_ = RobotCarDataset(ds_dir=ds_dir, train=True)
    test_ds_ = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=True)
    patchnetvlad_process(train_ds_, test_ds_)

    trainer_ = RobotCarTrainer(
        train_ds_,
        test_ds_,
        global_desc_dim,
        encoder_global,
        conf_ns_retrieval,
    )
    res_name = f"output/{trainer_.ds_name}/Robotcar_eval_{trainer_.global_desc_model_name}_{trainer_.global_feature_dim}.txt"
    trainer_.evaluate(res_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="../descriptor-disambiguation/datasets/robotcar",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--global_desc",
        type=str,
        default="mixvpr",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    run_function(
        args.dataset,
        args.global_desc,
        int(args.global_desc_dim),
    )
