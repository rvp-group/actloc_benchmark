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
    gt_poses, es_poses, cam_scale=0.4, cam_radius=0.025, with_coords=True,
    scale_sep=0.06,        # GT: (1 - scale_sep), EST: (1 + scale_sep)
    axial_nudge=0.01       # meters, EST nudged along its +Z
):
    """
    Always applies separation when both GT and EST are available at index i:
      - GT frustum scaled down, EST scaled up (concentric edges are visible).
    """
    gt_cam, est_cam = [], []

    def add_cam_and_frame(T, color, scale, radius, container):
        cam = camera_vis_with_cylinders(
            T, wh_ratio=4.0/3.0, scale=scale, fovx=90.0, color=color, radius=radius
        )
        if with_coords:
            cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
            cf.transform(T)
            container.append(cf)
        # small sphere at camera center
        s = o3d.geometry.TriangleMesh.create_sphere(radius=cam_radius*0.6)
        s.paint_uniform_color(color)
        s.translate(T[:3,3])
        container.extend(cam + [s])

    # how many rows to iterate (handle unequal lengths gracefully)
    N = 0
    if gt_poses is not None: N = max(N, gt_poses.shape[0])
    if es_poses is not None: N = max(N, es_poses.shape[0])

    for i in range(N):
        T_gt = gt_poses[i] if (gt_poses is not None and i < gt_poses.shape[0]) else None
        T_es = es_poses[i] if (es_poses is not None and i < es_poses.shape[0]) else None

        if T_gt is not None and T_es is not None:
            # scales for a matched pair
            gt_scale_i = cam_scale * (1.0 - scale_sep)
            es_scale_i = cam_scale * (1.0 + scale_sep)

            # nudge EST along its own +Z
            T_nudge = np.eye(4); T_nudge[:3,3] = np.array([0.0, 0.0, axial_nudge])
            T_es_draw = T_es @ T_nudge

            add_cam_and_frame(T_gt, (0,1,0), gt_scale_i, cam_radius*0.9, gt_cam)
            add_cam_and_frame(T_es_draw, (1,0,0), es_scale_i, cam_radius*1.1, est_cam)

        else:
            # draw whichever exists, no separation needed
            if T_gt is not None:
                add_cam_and_frame(T_gt, (0,1,0), cam_scale, cam_radius, gt_cam)
            if T_es is not None:
                add_cam_and_frame(T_es, (1,0,0), cam_scale, cam_radius, est_cam)

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


def visualize(geo_list, init_Twc=None, init_Tcw=None, width=1280, height=800):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualization",
                      width=width, height=height, visible=True)

    for i, g in enumerate(geo_list):
        vis.add_geometry(g, reset_bounding_box=(i == 0))

    vis.poll_events()
    vis.update_renderer()

    vc = vis.get_view_control()
    params = vc.convert_to_pinhole_camera_parameters()

    if init_Tcw is None and init_Twc is not None:
        init_Tcw = np.linalg.inv(init_Twc)

    if init_Tcw is not None:
        params.extrinsic = np.asarray(init_Tcw, dtype=np.float64)
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

    vis.run()
    vis.destroy_window()



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


    visualize([mesh] + gt_cam + est_cam + cam_links + waypoints_geo, init_Twc=np.identity(4),
              width=1280, height=800)


if __name__ == "__main__":
    main()
