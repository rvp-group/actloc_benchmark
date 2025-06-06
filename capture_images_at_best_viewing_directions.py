import os
import sys
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Camera and rendering parameters
H = 320
W = 320
focal = 277  # To ensure the FOV is 60
fx = focal
fy = focal
cx = W/2.0 - 0.5
cy = H/2.0 - 0.5

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
        raise ValueError(f"Number of waypoints ({waypoints.shape[0]}) != number of angles ({angles.shape[0]})")
    
    print(f"Loaded {waypoints.shape[0]} waypoints and corresponding angles")
    return waypoints, angles

def set_camera_to_best_viewpoint(vis, position, x_angle, y_angle):
    """
    Set the camera to a specific position and orient it according to the best viewing angles.
    
    Args:
        vis: Open3D visualizer
        position: Camera position [x, y, z]
        x_angle: X-axis rotation angle in degrees (elevation)
        y_angle: Y-axis rotation angle in degrees (azimuth)
    """
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    
    # Set the camera position
    # Start with the default camera orientation
    initial_extrinsic = np.array([[0, -1, 0, 0],   # Transform camera viewpoint 
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])
    
    # Extract initial rotation
    initial_rotation = initial_extrinsic[:3, :3]
    
    # Apply the best viewing angles as rotations
    # Convert degrees to radians
    rotation_x = R.from_euler('x', np.deg2rad(x_angle)).as_matrix()
    rotation_y = R.from_euler('y', np.deg2rad(y_angle)).as_matrix()
    
    # Combine rotations with the initial rotation
    # Apply rotations in the same order as during training data generation
    combined_rotation = rotation_x @ rotation_y @ initial_rotation
    new_translation = -combined_rotation @ position
    
    # Create new extrinsic matrix
    new_extrinsic = np.eye(4)
    new_extrinsic[:3, :3] = combined_rotation
    new_extrinsic[:3, 3] = new_translation
    
    # Apply the camera parameters
    cam_params.extrinsic = new_extrinsic
    success = ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    
    if not success:
        print(f"Warning: Failed to set camera parameters for position {position}")
    
    vis.update_renderer()
    vis.poll_events()
    
    return success

def capture_best_viewpoint_image(vis, waypoint_idx, position, x_angle, y_angle, output_folder):
    """
    Capture an image at the best viewpoint for a given waypoint.
    
    Args:
        vis: Open3D visualizer
        waypoint_idx: Index of the waypoint (for naming)
        position: Camera position [x, y, z]
        x_angle: X-axis rotation angle in degrees
        y_angle: Y-axis rotation angle in degrees
        output_folder: Folder to save the captured image
    """
    # Set camera to best viewpoint
    success = set_camera_to_best_viewpoint(vis, position, x_angle, y_angle)
    
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
    filename = f"waypoint_{waypoint_idx + 1:05d}_x{int(x_angle)}_y{int(y_angle)}.jpg"
    filepath = os.path.join(output_folder, filename)
    
    # Save the image
    o3d.io.write_image(filepath, o3d.geometry.Image(image))
    
    print(f"Saved: {filename}")
    return True

def save_camera_info(waypoints, angles, output_folder):
    """Save camera information for each captured image."""
    info_file = os.path.join(output_folder, "best_viewpoints_info.txt")
    
    with open(info_file, 'w') as f:
        f.write("# Waypoint_ID X_coord Y_coord Z_coord X_angle Y_angle Filename\n")
        for i, (waypoint, angle) in enumerate(zip(waypoints, angles)):
            x_pos, y_pos, z_pos = waypoint
            x_angle, y_angle = angle
            filename = f"waypoint_{i + 1:03d}_x{int(x_angle)}_y{int(y_angle)}.jpg"
            f.write(f"{i + 1} {x_pos:.6f} {y_pos:.6f} {z_pos:.6f} {x_angle:.1f} {y_angle:.1f} {filename}\n")
    
    print(f"Saved camera info to: {info_file}")

def main():
    parser = argparse.ArgumentParser(description="Capture images at best viewpoints for waypoints")
    
    # Arguments with default values
    parser.add_argument('--mesh-file', type=str, default='./example_data/yPKGKBCyYx8.glb',
                       help='Path to mesh file (.glb, .obj, .ply, etc.)')
    parser.add_argument('--waypoints-file', type=str, default='./example_data/sampled_viewpoints.txt',
                       help='Path to waypoints file (Nx3 coordinates)')
    parser.add_argument('--angles-file', type=str, default='./example_data/best_viewing_angles.txt',
                       help='Path to best viewing angles file (Nx2, x_angle y_angle)')
    parser.add_argument('--output-folder', type=str, default='./example_data/best_viewpoint_images',
                       help='Output folder for captured images (default: ./example_data/best_viewpoint_images)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    if not os.path.exists(args.waypoints_file):
        raise FileNotFoundError(f"Waypoints file not found: {args.waypoints_file}")
    if not os.path.exists(args.angles_file):
        raise FileNotFoundError(f"Angles file not found: {args.angles_file}")
    
    # Set up paths
    mesh_path = args.mesh_file
    waypoints_path = args.waypoints_file
    angles_path = args.angles_file
    output_folder = args.output_folder
    
    try:
        # Load data
        print("Loading mesh...")
        scene_mesh = load_mesh(mesh_path)
        
        if not scene_mesh or scene_mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        
        print("Loading waypoints and angles...")
        waypoints, angles = load_waypoints_and_angles(waypoints_path, angles_path)
        
        # Setup visualizer
        print("Setting up visualizer...")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=W, height=H, visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().light_on = False
        vis.add_geometry(scene_mesh, reset_bounding_box=True)
        
        # Set viewpoint control
        ctr = set_viewpoint_ctr(vis)
        param = ctr.convert_to_pinhole_camera_parameters()
        
        # Set camera intrinsics
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        
        print(f"Starting image capture for {len(waypoints)} waypoints...")
        
        # Process each waypoint
        successful_captures = 0
        for i, (waypoint, angle) in enumerate(zip(waypoints, angles)):
            x_angle, y_angle = angle
            
            # Skip if angles are NaN (failed inference)
            if np.isnan(x_angle) or np.isnan(y_angle):
                print(f"Waypoint {i + 1}: Skipping due to invalid angles (NaN)")
                continue
            
            print(f"Processing waypoint {i + 1}/{len(waypoints)}: "
                  f"pos={waypoint}, angles=({x_angle:.1f}°, {y_angle:.1f}°)")
            
            success = capture_best_viewpoint_image(
                vis, i, waypoint, x_angle, y_angle, output_folder
            )
            
            if success:
                successful_captures += 1
        
        # Save camera information
        save_camera_info(waypoints, angles, output_folder)
        
        print(f"\nCompleted! Successfully captured {successful_captures}/{len(waypoints)} images")
        print(f"Images saved to: {output_folder}")
        
        # Cleanup
        vis.clear_geometries()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()