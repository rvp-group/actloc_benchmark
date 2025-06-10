import open3d as o3d
import numpy as np
import argparse
from utils.vis_utils import camera_vis_with_cylinders, create_cylinder_between_points

def load_poses(gt_path=None, es_path=None):
    # Load or create dummy poses
    if gt_path:
        gt_poses = np.load(gt_path)
    else:
        gt_poses = np.array(
            [
                [1.50433051e-18, -1.00000000e00, 0.00000000e00, -1.69184173e-04],
                [0.00000000e00, 0.00000000e00, -1.00000000e00, 1.00000000e00],
                [1.00000000e00, 1.50433051e-18, -0.00000000e00, 2.94185517e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        gt_poses = np.linalg.inv(gt_poses).reshape(1, 4, 4)  # Reshape to (1, 4, 4)
    if es_path:
        es_poses = np.load(es_path)
    else:
        es_poses = gt_poses.copy()
        es_poses[:, :3, 3] += 0.5  # Shift estimated poses by 0.5 in x direction
    assert gt_poses.shape[1] == 4, "Ground truth poses should be of shape (N, 4, 4)"
    assert es_poses.shape[1] == 4, "Estimated poses should be of shape (N, 4, 4)"
    assert (
        gt_poses.shape == es_poses.shape
    ), "Ground truth and estimated poses should have the same shape"
    return gt_poses, es_poses


def create_camera_geometries(gt_poses, es_poses, cam_scale=0.4, cam_radius=0.025):
    gt_cam, est_cam = [], []
    for i in range(gt_poses.shape[0]):
        cam_1 = camera_vis_with_cylinders(
            gt_poses[
                i
            ],  # IMPORTANT: poses here should be camera in world (inverse of camera extrinsics)
            wh_ratio=4.0 / 3.0,
            scale=cam_scale,
            fovx=90.0,
            color=(0, 1, 0),
            radius=cam_radius,
        )
        cam_2 = camera_vis_with_cylinders(
            es_poses[i],
            wh_ratio=4.0 / 3.0,
            scale=cam_scale,
            fovx=90.0,
            color=(1, 0, 0),
            radius=cam_radius,
        )
        gt_cam.extend(cam_1)
        est_cam.extend(cam_2)
    return gt_cam, est_cam


def create_pose_links(gt_poses, es_poses):
    cam_links = []
    for i in range(gt_poses.shape[0]):
        try:
            gt_center = gt_poses[i][:3, 3]
            es_center = es_poses[i][:3, 3]
            cylinder = create_cylinder_between_points(
                gt_center, es_center, radius=0.03, color=(0, 0, 0)
            )
            cam_links.append(cylinder)
        except Exception as e:
            print(f"Error creating cylinder between poses {i}: {e}")
    return cam_links


def visualize(mesh, gt_cam, est_cam, cam_links):
    o3d.visualization.draw_geometries(
        [mesh] + gt_cam + est_cam + cam_links,
        window_name="Camera Frustum Visualization",
        mesh_show_back_face=False,
        mesh_show_wireframe=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a mesh and a cylinder between two points."
    )
    parser.add_argument(
        "--meshfile", type=str, required=True, help="Path to the mesh file."
    )
    parser.add_argument(
        "--gt_poses",
        type=str,
        required=False,
        help="Path to the ground truth poses npy file.",
    )
    parser.add_argument(
        "--es_poses",
        type=str,
        required=False,
        help="Path to the estimated poses npy file.",
    )
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.meshfile, enable_post_processing=True)
    gt_poses, es_poses = load_poses(args.gt_poses, args.es_poses)
    gt_cam, est_cam = create_camera_geometries(gt_poses, es_poses)
    cam_links = create_pose_links(gt_poses, es_poses)
    visualize(mesh, gt_cam, est_cam, cam_links)


if __name__ == "__main__":
    main()
