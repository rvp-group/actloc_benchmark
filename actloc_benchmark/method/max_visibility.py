import numpy as np
import logging
from scipy.spatial.transform import Rotation as Rot
from camera import Camera
cam = Camera()

def get_rotation_matrix_from_angles(azim, elev):
    R = Rot.from_euler('ZXZ', [azim, elev, 180], degrees=True)
    return R.as_matrix()

def project(intrinsics, R, t, points, min_depth=0, max_depth=3):
    # transform points to camera frame
    translated = points - t
    points_cam = (R.T @ translated.T).T  # shape (n, 3)
    # keep points in front of camera (z > 0)
    front_mask = points_cam[:, 2] > min_depth
    points_cam = points_cam[front_mask]
    
    # keep points maximum at very few meters, since we do not account for occlusions here
    far_mask = points_cam[:, 2] < max_depth
    points_cam = points_cam[far_mask]

    if points_cam.shape[0] == 0:
        return 0

    fx, fy, cx, cy, height, width = intrinsics

    # project to image plane
    u = fx * (points_cam[:, 0] / points_cam[:, 2]) + cx
    v = fy * (points_cam[:, 1] / points_cam[:, 2]) + cy

    # valid projection bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    return in_bounds, np.int32(u+0.5), np.int32(v+0.5), points_cam[:, 2]

def count_visible_points(waypoint, R, points, intrinsics):
    in_bounds, _, _, _ = project(intrinsics, R, waypoint, points)
    return np.count_nonzero(in_bounds)

def predict_pose(waypoint: np.ndarray, waypoint_idx: int, points: np.ndarray):
    """run inference for a single waypoint"""
    logging.info(f"processing waypoint {waypoint_idx}: {waypoint}")
    
    # x-axis (elevation): 6 cells covering [-60, 40] with interval 20
    # y-axis (azimuth): 18 cells covering [-180, 160] with interval 20
    x_angles = np.arange(-60, 60, 20)  # [-60, -40, -20, 0, 20, 40] (6 values)
    y_angles = np.arange(-180, 180, 20)  # [-180, -160, ..., 160] (18 values)

    # main loop to compute visibility per view direction
    visible_counts = np.zeros((len(x_angles), len(y_angles)), dtype=int)

    for i, elev in enumerate(x_angles):
        for j, azim in enumerate(y_angles):
            R = get_rotation_matrix_from_angles(azim, elev)
            count = count_visible_points(
                waypoint, R, points, [cam.fx, cam.fy, cam.cx, cam.cy, cam.H, cam.W]
            )
            visible_counts[i, j] = count

    best_idx = np.unravel_index(np.argmax(visible_counts), visible_counts.shape)
    
    best_elev = x_angles[best_idx[0]]
    best_azim = y_angles[best_idx[1]]
    
    R = get_rotation_matrix_from_angles(best_elev, best_azim)
    
    return Rot.from_matrix(R).as_quat()


def filter_points_by_error(points3D: dict, error_threshold: float = 0.5):
    """filter 3d points by reprojection error"""
    filtered_points = []

    for point_id, pt in points3D.items():
        if pt.error < error_threshold:
            filtered_points.append(pt.xyz)

    if not filtered_points:
        raise ValueError(
            f"no points remain after filtering with error threshold {error_threshold}"
        )

    points_array = np.array(filtered_points, dtype=np.float32)

    logging.info(
        f"filtered points: kept {len(points_array)} out of {len(points3D)} points (error < {error_threshold})"
    )
    return points_array
