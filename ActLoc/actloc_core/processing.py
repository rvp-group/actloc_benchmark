import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from ActLoc.utils.read_write_model import qvec2rotmat

def filter_points_by_error(points3D: dict, error_threshold: float = 0.5):
    """Filter 3D points by reprojection error."""
    filtered_points = []
    filtered_colors = []
    
    for _, pt in points3D.items():
        if pt.error < error_threshold:
            filtered_points.append(pt.xyz)
            filtered_colors.append(pt.rgb)

    if len(filtered_points) == 0:
        raise ValueError(f"No points remain after filtering with error threshold {error_threshold}")
    
    points_array = np.array(filtered_points, dtype=np.float32)
    colors_array = np.array(filtered_colors, dtype=np.uint8)
    
    logging.info(f"Filtered points: kept {len(points_array)} out of {len(points3D)} points (error < {error_threshold})")
    return points_array, colors_array

def transform_data(points: np.ndarray, colors: np.ndarray, images: dict, new_origin: np.ndarray):
    """Transform point cloud and camera poses relative to new origin."""
    # Transform points
    transformed_points = points - new_origin
    transformed_colors = colors.copy()
    
    # Transform camera poses
    rotmats_list = []
    camera_centers_list = []
    
    for img in images.values():
        R_mat = qvec2rotmat(img.qvec)
        # Original camera center: C = -R^T * tvec
        C = -R_mat.T @ img.tvec
        C_new = C - new_origin  # New camera center
        
        rotmats_list.append(R_mat)
        camera_centers_list.append(C_new)
    
    rotmats = np.array(rotmats_list)
    camera_centers = np.array(camera_centers_list)
    
    return transformed_points, transformed_colors, rotmats, camera_centers

def crop_to_bounding_box(points: np.ndarray, colors: np.ndarray,
                        x_range=(-4, 4), y_range=(-4, 4), z_range=(-2, 2)):
    """Crop points and colors to specified bounding box."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    
    cropped_points = points[mask]
    cropped_colors = colors[mask]
    
    if len(cropped_points) == 0:
        raise ValueError("No points remain after bounding box cropping")
    
    logging.info(f"Cropped points: {len(points)} -> {len(cropped_points)} points")
    return cropped_points, cropped_colors

def prepare_features_for_one_waypoint(points: np.ndarray, colors: np.ndarray, rotmats: np.ndarray, camera_centers: np.ndarray):
    """Prepare features for model input."""
    # Point cloud features: [x, y, z, r, g, b] (normalize colors to [0,1])
    pc_features = np.hstack([
        points.astype(np.float32),
        colors.astype(np.float32) / 255.0
    ])
    
    # Convert rotation matrices to quaternions (w, x, y, z format - scalar first)
    quats_list = []
    for R_mat in rotmats:
        try:
            quat = R.from_matrix(R_mat).as_quat(canonical=True, scalar_first=True)  # [w, x, y, z]
            quats_list.append(quat)
        except ValueError:
            logging.warning("Invalid rotation matrix, using identity quaternion")
            quats_list.append(np.array([1.0, 0.0, 0.0, 0.0]))  # Identity quaternion [w, x, y, z]
    
    camera_quats = np.array(quats_list, dtype=np.float32)
    
    # Camera pose features: [cx, cy, cz, qw, qx, qy, qz]
    pose_features = np.hstack([
        camera_centers.astype(np.float32),
        camera_quats
    ])
    
    logging.info(f"Prepared features - PC: {pc_features.shape}, Pose: {pose_features.shape}")
    return pc_features, pose_features

BASE_ORIENTATION = np.array([[0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]])

def calculate_extrinsic_matrix(position: np.ndarray, x_angle: float, y_angle: float) -> np.ndarray:
    """
    Calculates the 4x4 camera extrinsic matrix from a position and relative viewing angles.
    
    Args:
        position: Camera position [x, y, z] in world coordinates.
        x_angle: X-axis rotation angle in degrees (elevation).
        y_angle: Y-axis rotation angle in degrees (azimuth).
        
    Returns:
        A 4x4 numpy array representing the camera extrinsic matrix.
    """
    initial_rotation = BASE_ORIENTATION[:3, :3]
    
    # Apply the best viewing angles as rotations
    rotation_x = R.from_euler('x', np.deg2rad(x_angle)).as_matrix()
    rotation_y = R.from_euler('y', np.deg2rad(y_angle)).as_matrix()
    
    # Combine rotations with the initial rotation
    combined_rotation = rotation_x @ rotation_y @ initial_rotation
    
    # Calculate the translation part of the extrinsic matrix (-R*t)
    new_translation = -combined_rotation @ position
    
    # Create new extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = combined_rotation
    extrinsic[:3, 3] = new_translation
    
    return extrinsic