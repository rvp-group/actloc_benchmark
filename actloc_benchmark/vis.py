import open3d as o3d
import numpy as np
import argparse
from utils.vis_utils import camera_vis_with_cylinders, create_cylinder_between_points
from utils.io import load_waypoints, parse_poses_file


def load_poses(gt_path=None, es_path=None):
    # Load or create dummy poses
    if gt_path:
        gt_poses = parse_poses_file(gt_path, Twc=True)
        gt_poses = np.array([Twc for Twc in gt_poses.values()])

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
        es_poses = parse_poses_file(es_path, Twc=True)
        es_poses = np.array([Twc for Twc in es_poses.values()])
    else:
        es_poses = gt_poses.copy()
        es_poses[:, :3, 3] += 0.5  # Shift estimated poses by 0.5 in x direction
    assert gt_poses.shape[1] == 4, "Ground truth poses should be of shape (N, 4, 4)"
    assert es_poses.shape[1] == 4, "Estimated poses should be of shape (N, 4, 4)"
    assert (
        gt_poses.shape == es_poses.shape
    ), "Ground truth and estimated poses should have the same shape"

    return gt_poses, es_poses


def create_camera_geometries(
    gt_poses, es_poses, cam_scale=0.4, cam_radius=0.025, with_coords=True
):
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

        if with_coords:
            # Add coordinate frames for each camera
            coord_frame_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0]
            )
            coord_frame_gt.transform(gt_poses[i])
            gt_cam.append(coord_frame_gt)
            coord_frame_es = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0]
            )
            coord_frame_es.transform(es_poses[i])
            est_cam.append(coord_frame_es)

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


def create_waypoint_geometries(waypoints, radius=0.02, color=(0, 0, 1)):
    # create spheres at each waypoint
    waypoints_geo = []
    for wp in waypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(wp)
        sphere.paint_uniform_color(color)
        waypoints_geo.append(sphere)
    return waypoints_geo


def visualize(geo_list):
    o3d.visualization.draw_geometries(
        geo_list,
        window_name="Camera Frustum Visualization",
        mesh_show_back_face=False,
        mesh_show_wireframe=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a mesh and a cylinder between two points."
    )
    parser.add_argument(
        "--meshfile",
        type=str,
        required=False,
        help="Path to the mesh file.",
        default="./example_data/00005-yPKGKBCyYx8/yPKGKBCyYx8.glb",
    )
    parser.add_argument(
        "--gt_poses",
        type=str,
        required=False,
        help="Path to the ground truth poses txt file.",
    )
    parser.add_argument(
        "--es_poses",
        type=str,
        required=False,
        help="Path to the estimated poses txt file.",
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help="Path to sampled waypoints txt file.",
    )

    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.meshfile, enable_post_processing=True)
    gt_poses, es_poses = load_poses(args.gt_poses, args.es_poses)
    gt_cam, est_cam = create_camera_geometries(gt_poses, es_poses)
    cam_links = create_pose_links(gt_poses, es_poses)

    waypoints_geo = []
    if args.waypoints:
        waypoints = load_waypoints(args.waypoints)
        waypoints = np.array([wp for wp in waypoints.values()])
        waypoints_geo.extend(
            create_waypoint_geometries(waypoints, radius=0.2, color=(0, 0, 1))
        )

    visualize([mesh] + gt_cam + est_cam + cam_links + waypoints_geo)


if __name__ == "__main__":
    main()
