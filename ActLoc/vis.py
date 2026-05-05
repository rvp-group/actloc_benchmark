import open3d as o3d
import numpy as np
import argparse
from utils.vis_utils import camera_vis_with_cylinders, create_waypoint_geometries
from utils.io import load_waypoints, load_predicted_poses

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
        "--poses-file",
        type=str,
        required=False,
        default=None,
        help="Path to the selected poses txt file.",
    )
    parser.add_argument(
        "--waypoints-file",
        type=str,
        default=None,
        required=False,
        help="Path to sampled waypoints txt file.",
    )

    args = parser.parse_args()
    mesh = o3d.io.read_triangle_mesh(args.meshfile, enable_post_processing=True)

    waypoints_geo = []
    if args.waypoints_file:
        waypoints = load_waypoints(args.waypoints_file)
        if waypoints.shape[0] > 0:
            waypoints_geo.extend(
                create_waypoint_geometries(waypoints, radius=0.1, color=(0, 0, 1))
            )

    cam_geo = []
    if args.poses_file:
        _,_,se_poses = load_predicted_poses(args.poses_file)
        for i in range(se_poses.shape[0]):
            cam_geo.append(camera_vis_with_cylinders(
                np.linalg.inv(se_poses[i]),
                wh_ratio=4.0 / 3.0,
                scale=0.4,
                fovx=90.0,
                color=(1.0, 0.8667, 0.0),  # yellow
                radius=0.025,
                return_mesh=True,
            ))

    observer_Twc = np.array([
        [-2.19651623e-03, -9.99937606e-01, -1.09525786e-02,  0.0],
        [-9.99869439e-01,  2.02076514e-03,  1.60318772e-02,  0.0],
        [-1.60087443e-02,  1.09863629e-02, -9.99811492e-01,  5.0],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ], dtype=np.float64)

    visualize([mesh] + cam_geo + waypoints_geo, init_Twc=observer_Twc,
              width=1280, height=1000)


if __name__ == "__main__":
    main()
