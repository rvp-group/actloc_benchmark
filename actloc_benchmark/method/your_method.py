import logging
import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import torch

# Ensure repo root on sys.path for "ActLoc" imports (run.sh changes cwd)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ActLoc pipeline imports
from ActLoc.actloc_core.processing import (
    filter_points_by_error as actloc_filter_points_by_error,
    calculate_extrinsic_matrix,
)
from ActLoc.actloc_core.torch_utils import load_model
from ActLoc.utils.helpers import run_inference_for_one_waypoint


def get_orientations_by_nearest_neighbor(images: dict, waypoints: dict):
    """
    For each waypoint, finds the orientation of the closest original camera
    from the sparse reconstruction.
    Returns a dict[id] -> quaternion (x, y, z, w).
    """
    # Collect COLMAP camera poses
    sorted_image_ids = sorted(images.keys())
    original_qvecs = np.array([images[i].qvec for i in sorted_image_ids])
    original_tvecs = np.array([images[i].tvec for i in sorted_image_ids])

    # KD-tree over camera centers
    kdtree = KDTree(original_tvecs)

    best_angles = {}
    for wid, waypoint_pos in waypoints.items():
        # Query the nearest SfM camera
        _, nearest_index = kdtree.query(waypoint_pos)
        colmap_qwxyz = original_qvecs[nearest_index]
        # Convert (qw,qx,qy,qz) -> (x,y,z,w)
        quat_xyzw = np.array([colmap_qwxyz[1], colmap_qwxyz[2], colmap_qwxyz[3], colmap_qwxyz[0]], dtype=np.float64)
        best_angles[wid] = quat_xyzw
    return best_angles


# ActLoc model-backed predictor
_DEFAULT_ACTLOC_CKPT = os.path.join(ROOT_DIR, "ActLoc", "checkpoints", "trained_actloc.pth")
_ACTLOC_MODEL = None
_ACTLOC_DEVICE = None
_USE_BF16 = False
_AMP_ENABLED = True


def _ensure_actloc_model(checkpoint_path: str):
    """
    Lazy-load and cache ActLoc model on CUDA. If checkpoint missing or CUDA unavailable,
    return (None, None, False, False) to trigger fallback.
    """
    global _ACTLOC_MODEL, _ACTLOC_DEVICE, _USE_BF16, _AMP_ENABLED

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT_DIR, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        logging.warning(f"ActLoc checkpoint not found: {checkpoint_path}")
        return None, None, False, False

    if _ACTLOC_MODEL is None:
        if not torch.cuda.is_available():
            logging.warning("CUDA not available; falling back to nearest-neighbor orientation.")
            return None, None, False, False
        _ACTLOC_DEVICE = torch.device("cuda")
        _USE_BF16 = torch.cuda.is_bf16_supported()
        _ACTLOC_MODEL = load_model(checkpoint_path, _ACTLOC_DEVICE)
        _ACTLOC_MODEL.eval()
    return _ACTLOC_MODEL, _ACTLOC_DEVICE, _USE_BF16, _AMP_ENABLED


def actloc_predict_best_angles(input_dict: dict, checkpoint_path: str = _DEFAULT_ACTLOC_CKPT) -> dict:
    """
    Predict best orientation(s) at given waypoint(s) using ActLoc.
    Falls back to nearest-neighbor COLMAP orientation if checkpoint/GPU missing.
    Returns dict[id] -> quaternion (x,y,z,w).
    """
    images = input_dict["images"]
    points3D = input_dict["points3D"]
    waypoints = input_dict["waypoints"]

    # Prepare 3D inputs (ActLoc preprocessing)
    points, colors = actloc_filter_points_by_error(points3D)

    model, device, use_bf16, amp_enabled = _ensure_actloc_model(checkpoint_path)
    if model is None:
        return get_orientations_by_nearest_neighbor(images, waypoints)

    results = {}
    for idx, (wid, waypoint) in enumerate(waypoints.items()):
        try:
            res = run_inference_for_one_waypoint(
                waypoint=np.asarray(waypoint, dtype=np.float32),
                waypoint_idx=idx,
                filtered_points=points,
                filtered_colors=colors,
                images=images,
                model=model,
                device=device,
                amp_enabled=amp_enabled,
                use_bf16=use_bf16,
            )
            if res is None:
                raise RuntimeError("ActLoc returned None")
            _, _, best_dir = res
            x_angle = float(best_dir["best_x_angle"])
            y_angle = float(best_dir["best_y_angle"])
            extrinsic = calculate_extrinsic_matrix(
                position=np.asarray(waypoint, dtype=np.float32),
                x_angle=x_angle,
                y_angle=y_angle,
            )
            Rcw = extrinsic[:3, :3]
            Rwc = Rcw.T
            quat_xyzw = R.from_matrix(Rwc).as_quat()
            results[wid] = quat_xyzw.astype(np.float64)
        except Exception as e:
            logging.warning(f"ActLoc inference failed for waypoint {wid}: {e}; using nearest neighbor.")
            nn = get_orientations_by_nearest_neighbor(images, {wid: waypoint})
            results[wid] = nn[wid]
    return results