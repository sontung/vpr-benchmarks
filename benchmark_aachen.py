import argparse

import numpy as np
import benchmark_utils
from dataset import AachenDataset
from trainers import BaseTrainer


def run_function(
    ds_dir,
    retrieval_model,
    global_desc_dim,
):
    print(f"Using {retrieval_model}-{global_desc_dim}")

    encoder_global, conf_ns_retrieval = benchmark_utils.prepare_encoders(
        retrieval_model, global_desc_dim
    )
    train_ds_ = AachenDataset(ds_dir=ds_dir)
    test_ds_ = AachenDataset(ds_dir=ds_dir, train=False)
    trainer_ = BaseTrainer(
        train_ds_,
        test_ds_,
        global_desc_dim,
        encoder_global,
        conf_ns_retrieval,
    )
    res_name = f"output/{trainer_.ds_name}/Aachen_v1_1_eval_{trainer_.global_desc_model_name}_{trainer_.global_feature_dim}.txt"

    trainer_.evaluate(res_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="../descriptor-disambiguation/datasets/aachen_v1.1",
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
