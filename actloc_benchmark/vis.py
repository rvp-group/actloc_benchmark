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
        assert gt_poses.shape[1] == 4, "Ground truth poses should be of shape (N, 4, 4)"

    else:
        gt_poses = None

    if es_path:
        es_poses = parse_poses_file(es_path, Twc=True)
        es_poses = np.array([Twc for Twc in es_poses.values()])
        assert es_poses.shape[1] == 4, "Estimated poses should be of shape (N, 4, 4)"
    else:
        es_poses = None

    return gt_poses, es_poses


def create_camera_geometries(
    gt_poses, es_poses, cam_scale=0.4, cam_radius=0.025, with_coords=True
):
    gt_cam, est_cam = [], []

    if gt_poses is not None:
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

            if with_coords:
                # Add coordinate frames for each camera
                coord_frame_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1, origin=[0, 0, 0]
                )
                coord_frame_gt.transform(gt_poses[i])
                gt_cam.append(coord_frame_gt)

            gt_cam.extend(cam_1)

    if es_poses is not None:
        for i in range(es_poses.shape[0]):
            cam_2 = camera_vis_with_cylinders(
                es_poses[i],
                wh_ratio=4.0 / 3.0,
                scale=cam_scale,
                fovx=90.0,
                color=(1, 0, 0),
                radius=cam_radius,
            )

            if with_coords:
                # Add coordinate frames for each camera
                coord_frame_es = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1, origin=[0, 0, 0]
                )
                coord_frame_es.transform(es_poses[i])
                est_cam.append(coord_frame_es)

        est_cam.extend(cam_2)

    return gt_cam, est_cam


def create_pose_links(gt_poses, es_poses):
    cam_links = []
    if gt_poses is None or es_poses is None:
        print("No poses provided, skipping pose links creation.")
        return cam_links
    if gt_poses.shape[0] != es_poses.shape[0]:
        print(
            "Ground truth and estimated poses have different lengths, cannot create links."
        )
        return cam_links
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
        required=True,
        help="Path to the mesh file.",
        default="./example_data/00010-DBjEcHFg4oq/DBjEcHFg4oq.glb",
    )
    parser.add_argument(
        "--gt_poses",
        type=str,
        required=False,
        default=None,
        help="Path to the ground truth poses txt file.",
    )
    parser.add_argument(
        "--es_poses",
        type=str,
        required=False,
        default=None,
        help="Path to the estimated poses txt file.",
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        required=False,
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
        if len(waypoints) > 0:
            waypoints = np.array([wp for wp in waypoints.values()])
            waypoints_geo.extend(
                create_waypoint_geometries(waypoints, radius=0.1, color=(0, 0, 1))
            )

    visualize([mesh] + gt_cam + est_cam + cam_links + waypoints_geo)


if __name__ == "__main__":
    main()
