import numpy as np
import open3d as o3d
import os
import logging

def load_mesh(meshfile):
    """Load a mesh from file."""
    mesh = o3d.io.read_triangle_mesh(meshfile, enable_post_processing=True)
    print("Mesh loaded: {}".format(mesh))
    return mesh

def load_waypoints(waypoints_file: str) -> np.ndarray:
    """Load waypoint coordinates from text file."""
    if not os.path.exists(waypoints_file):
        raise FileNotFoundError(f"Waypoints file not found: {waypoints_file}")
    
    waypoints = np.loadtxt(waypoints_file, dtype=np.float32)
    if waypoints.ndim == 1 and waypoints.shape[0] == 3:
        waypoints = waypoints.reshape(1, 3)
    elif waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"Expected waypoints shape (N, 3), got {waypoints.shape}")
    
    logging.info(f"Loaded {waypoints.shape[0]} waypoints from {waypoints_file}")
    return waypoints

def load_predicted_poses(npz_file):
    """Load predicted poses from an .npz file."""
    data = np.load(npz_file)
    return data['waypoints'], data['angles'], data['extrinsics']

def save_camera_info(waypoints, angles, output_folder):
    """Save camera information for each captured image."""
    info_file = os.path.join(output_folder, "best_viewpoints_info.txt")
    
    with open(info_file, 'w') as f:
        f.write("# Waypoint_ID X_coord Y_coord Z_coord X_angle Y_angle Filename\n")
        for i, (waypoint, angle) in enumerate(zip(waypoints, angles)):
            x_pos, y_pos, z_pos = waypoint
            x_angle, y_angle = angle
            filename = f"waypoint_{i + 1:05d}_x{int(x_angle)}_y{int(y_angle)}.jpg"
            f.write(f"{i + 1} {x_pos:.6f} {y_pos:.6f} {z_pos:.6f} {x_angle:.1f} {y_angle:.1f} {filename}\n")
    
    print(f"Saved camera info to: {info_file}")

