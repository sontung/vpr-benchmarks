import argparse

import benchmark_utils
from dataset import CMUDataset
from trainers import BaseTrainer

TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def run_function(
    ds_dir,
    retrieval_model,
    global_desc_dim,
):
    encoder_global, conf_ns_retrieval = benchmark_utils.prepare_encoders(
        retrieval_model, global_desc_dim
    )
    results = []
    for slice_ in TEST_SLICES:
        print(f"Processing slice {slice_}")
        train_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}")
        test_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}", train=False)

        trainer_ = BaseTrainer(
            train_ds_,
            test_ds_,
            global_desc_dim,
            encoder_global,
            conf_ns_retrieval,
        )
        query_results = trainer_.evaluate(f"output/cmu/{slice_}.txt", return_results=True)
        results.extend(query_results)
        del trainer_

    result_file = open(
        f"output/cmu/CMU_eval_{retrieval_model}_{global_desc_dim}.txt",
        "w",
    )
    for line in results:
        print(line, file=result_file)
    result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/datasets/cmu_extended",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--global_desc",
        type=str,
        default="eigenplaces",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=2048,
    )
    args = parser.parse_args()

    run_function(
        args.dataset,
        args.global_desc,
        int(args.global_desc_dim),
    )
