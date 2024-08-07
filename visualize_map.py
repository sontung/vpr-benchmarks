import numpy as np
import open3d as o3d

import benchmark_utils
from dataset import RobotCarDataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def read_poses_from_file(file_name="/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/Robotcar_eval_d2net_matches-NN-mutual_pairs-query-salad-10.txt"):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    poses = []
    for line in lines:
        _, qw, qx, qy, qz, tx, ty, tz = line.split(" ")
        qw, qx, qy, qz, tx, ty, tz = map(float, [qw, qx, qy, qz, tx, ty, tz])
        poses.append([qw, qx, qy, qz, tx, ty, tz])
    return poses


def main():
    ds_dir = "../descriptor-disambiguation/datasets/robotcar"
    intrinsics = np.eye(3)

    intrinsics[0, 0] = 738
    intrinsics[1, 1] = 738
    intrinsics[0, 2] = 427  # 427
    intrinsics[1, 2] = 240
    train_ds_ = RobotCarDataset(ds_dir=ds_dir, train=True)

    xyz_arr = benchmark_utils.read_nvm_file(train_ds_.sfm_model_dir, read_xyz=True)

    index_map_file_name = f"output/robotcar/indices.npy"
    inlier_ind = np.load(index_map_file_name)

    # if os.path.isfile(index_map_file_name):
    #     inlier_ind = np.load(index_map_file_name)
    # else:
    #     import open3d as o3d
    #
    #     point_cloud = o3d.geometry.PointCloud(
    #         o3d.utility.Vector3dVector(xyz_arr)
    #     )
    #     cl, inlier_ind = point_cloud.remove_radius_outlier(
    #         nb_points=16, radius=5, print_progress=True
    #     )
    #
    #     np.save(index_map_file_name, np.array(inlier_ind))
    test_ds_ = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=False)
    predicted_poses = read_poses_from_file("/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/Robotcar_eval_d2net_nn.txt")
    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr[inlier_ind]))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 0.5
    # vis.add_geometry(pc1)
    for idx in range(0, len(train_ds_), 50):
        example = train_ds_[idx]
        qw, qx, qy, qz, tx, ty, tz = example[2]
        # tx, ty, tz = -(Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ np.array([tx, ty, tz]))
        #
        pose_mat = benchmark_utils.return_pose_mat_no_inv(
            [qw, qx, qy, qz], [tx, ty, tz]
        )
        cam = o3d.geometry.LineSet.create_camera_visualization(
            427 * 2, 240 * 2, intrinsics, pose_mat, scale=10
        )
        vis.add_geometry(cam)
    for idx in range(0, len(predicted_poses), 1):
        qw, qx, qy, qz, tx, ty, tz = predicted_poses[idx]
        pose_mat = benchmark_utils.return_pose_mat_no_inv(
            [qw, qx, qy, qz], [tx, ty, tz]
        )
        cam = o3d.geometry.LineSet.create_camera_visualization(
            427 * 2, 240 * 2, intrinsics, pose_mat, scale=10
        )
        vis.add_geometry(cam)
        cam.paint_uniform_color((0, 1, 0))

    for idx in range(0, len(test_ds_), 1):
        example = test_ds_[idx]
        qw, qx, qy, qz, tx, ty, tz = example[2]
        # tx, ty, tz = -(Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ np.array([tx, ty, tz]))
        #
        pose_mat = benchmark_utils.return_pose_mat_no_inv(
            [qw, qx, qy, qz], [tx, ty, tz]
        )
        cam = o3d.geometry.LineSet.create_camera_visualization(
            427 * 2, 240 * 2, intrinsics, pose_mat, scale=10
        )
        cam.paint_uniform_color((1, 0, 0))
        vis.add_geometry(cam)
    vis.run()
    vis.destroy_window()
    return


if __name__ == '__main__':
    main()
