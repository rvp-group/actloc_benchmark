import os
import sys
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from utils.io import *


from camera import Camera

cam = Camera()  # instantiate one for all


def load_mesh(meshfile):
    """Load a mesh from file."""
    mesh = o3d.io.read_triangle_mesh(meshfile, enable_post_processing=True)
    print("Mesh loaded: {}".format(mesh))
    return mesh


def set_viewpoint_ctr(vis):
    """Set the viewpoint control and configure the near and far clipping planes."""
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(40.0)  # Far clipping plane at 40 units
    ctr.set_constant_z_near(0.02)  # Near clipping plane at 0.02 units
    return ctr


def load_waypoints_and_angles(waypoints_file, angles_file):
    """Load waypoint coordinates and corresponding best viewing angles."""
    waypoints = np.loadtxt(waypoints_file, dtype=np.float32)
    if waypoints.ndim == 1 and waypoints.shape[0] == 3:
        waypoints = waypoints.reshape(1, 3)
    elif waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"Expected waypoints shape (N, 3), got {waypoints.shape}")

    angles = np.loadtxt(angles_file, dtype=np.float32)
    if angles.ndim == 1 and angles.shape[0] == 2:
        angles = angles.reshape(1, 2)
    elif angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError(f"Expected angles shape (N, 2), got {angles.shape}")

    if waypoints.shape[0] != angles.shape[0]:
        raise ValueError(
            f"Number of waypoints ({waypoints.shape[0]}) != number of angles ({angles.shape[0]})"
        )

    print(f"Loaded {waypoints.shape[0]} waypoints and corresponding angles")
    return waypoints, angles


def set_camera_to_best_viewpoint(vis, Twc):
    """
    Set the camera to a specific position and orient it according to the best viewing angles.

    Args:
        vis: Open3D visualizer
        Twc: homogenous camera matrix camera in world
       
    """

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Apply the camera parameters
    cam_params.extrinsic = np.linalg.inv(Twc)
    success = ctr.convert_from_pinhole_camera_parameters(
        cam_params, allow_arbitrary=True
    )

    if not success:
        print(f"Warning: Failed to set camera parameters for position {Twc[0:3, 3]}")

    vis.update_renderer()
    vis.poll_events()

    return success


def capture_best_viewpoint_image(
    vis, waypoint_idx, Twc, output_folder
):
    """
    Capture an image at the best viewpoint for a given waypoint.

    Args:
        vis: Open3D visualizer
        waypoint_idx: Index of the waypoint (for naming)
        Twc: 4x4 homogenous transformation describing standard cam in world
        output_folder: Folder to save the captured image
    """
    # Set camera to best viewpoint
    success = set_camera_to_best_viewpoint(vis, Twc)

    if not success:
        print(f"Failed to set camera for waypoint {waypoint_idx + 1}")
        return False

    # Capture the image
    image = vis.capture_screen_float_buffer(True)
    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate filename
    filename = f"{waypoint_idx}.png"
    filepath = os.path.join(output_folder, filename)

    # Save the image
    o3d.io.write_image(filepath, o3d.geometry.Image(image))

    print(f"\tsaved: {filename}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Capture images at best viewpoints for waypoints"
    )

    # Arguments with default values
    parser.add_argument(
        "--mesh-file",
        type=str,
        default="./example_data/yPKGKBCyYx8.glb",
        help="Path to mesh file (.glb, .obj, .ply, etc.)",
    )
    parser.add_argument(
        "--pose-file",
        type=str,
        default="./example_data/estimate/pose_estimate.txt",
        help="Path to pose estimate, this should be waypoints and best orientations",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="./example_data/estimate/best_viewpoint_images",
        help="Output folder for captured images (default: ./example_data/best_viewpoint_images)",
    )

    args = parser.parse_args()

    # validate input files
    if not os.path.exists(args.mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    if not os.path.exists(args.pose_file):
        raise FileNotFoundError(f"Pose estimate not found: {args.pose_file}")

    output_dir = os.path.dirname(args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # set up paths
    mesh_path = args.mesh_file
    poses_path = args.pose_file
    output_folder = args.output_folder

    try:
        # Load data
        print("Loading mesh...")
        scene_mesh = load_mesh(mesh_path)

        if not scene_mesh or scene_mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")

        print("Loading waypoints and angles...")
        poses = parse_poses_file(poses_path, Twc=True) # we want standard camera transform camera in world

        # Setup visualizer
        print("Setting up visualizer...")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=cam.W, height=cam.H, visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().light_on = False
        vis.add_geometry(scene_mesh, reset_bounding_box=True)

        # Set viewpoint control
        ctr = set_viewpoint_ctr(vis)
        param = ctr.convert_to_pinhole_camera_parameters()

        # Set camera intrinsics
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            cam.W, cam.H, cam.fx, cam.fy, cam.cx, cam.cy
        )
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        print(f"Starting image capture for {len(poses)} waypoints...")

        # Process each waypoint
        successful_captures = 0
        for idx, Twc in poses.items():
      
            print(f"Processing waypoint {idx}")

            success = capture_best_viewpoint_image(
                vis, idx, Twc, output_folder
            )

            if success:
                successful_captures += 1

        print(
            f"\nCompleted! Successfully captured {successful_captures}/{len(poses.keys())} images"
        )
        print(f"Images saved to: {output_folder}")

        # Cleanup
        vis.clear_geometries()
        vis.destroy_window()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
