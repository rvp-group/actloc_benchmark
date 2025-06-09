import numpy as np
import logging

fx, fy, cx, cy, height, width = [320, 320, 240, 320, 480, 640]


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_rotation_matrix(azimuth_deg, elevation_deg):
    # convert angles to radians
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # rotation around y (azimuth)
    R_az = np.array(
        [[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]]
    )

    # rotation around x (elevation)
    R_el = np.array(
        [[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]]
    )

    # final rotation matrix (camera looking in z after rotation)
    return R_el @ R_az


def count_visible_points(waypoint, R, points, intrinsics):
    # transform points to camera frame
    translated = points - waypoint
    points_cam = (R.T @ translated.T).T  # shape (n, 3)

    # keep points in front of camera (z > 0)
    front_mask = points_cam[:, 2] > 0
    points_cam = points_cam[front_mask]

    if points_cam.shape[0] == 0:
        return 0

    fx, fy, cx, cy, height, width = intrinsics

    # project to image plane
    u = fx * (points_cam[:, 0] / points_cam[:, 2]) + cx
    v = fy * (points_cam[:, 1] / points_cam[:, 2]) + cy

    # valid projection bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)

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
            R = get_rotation_matrix(azim, elev)
            count = count_visible_points(
                waypoint, R, points, [fx, fy, cx, cy, height, width]
            )
            visible_counts[i, j] = count

    best_idx = np.unravel_index(np.argmax(visible_counts), visible_counts.shape)
    best_elev = x_angles[best_idx[0]]
    best_azim = y_angles[best_idx[1]]

    return best_elev, best_azim


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
