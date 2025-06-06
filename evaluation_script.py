"""
ActLoc Model Evaluation Script

Evaluates a trained ActLoc model by:
1. Loading scenes from the new data structure
2. Optionally removing random percentages of cameras/images and associated points
3. Preprocessing data on-the-fly 
4. Finding best viewing directions for each waypoint
5. Checking if GT localization errors at best views fall within thresholds
6. Reporting threshold performance percentages for different SfM sparsification levels
"""

import argparse
import logging
import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from typing import Dict, Tuple
import random
import copy

try:
    from sample_data_parser import ActLocDataParser
    from utils.read_write_model import qvec2rotmat
    from models.new_model import TwoHeadTransformer
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Ensure data_parser.py, utils/read_write_model.py and models/new_model.py are accessible")
    sys.exit(1)

from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Default Model Parameters ---
DEFAULT_MODEL_PARAMS = {
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_cross_layers": 4,
    "dropout": 0.1,
    "dim_feedforward": 256 * 4,
}

def remove_cameras_and_associated_points(cameras: Dict, images: Dict, points3D: Dict, 
                                        removal_percentage: float, seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Remove a percentage of cameras/images and ALL points that are visible in ANY of those cameras.
    Uses reference script logic: image.point3D_ids to find points observed by deleted images (aggressive deletion).
    
    Args:
        cameras: Dictionary of camera objects
        images: Dictionary of image objects  
        points3D: Dictionary of 3D point objects
        removal_percentage: Percentage (0-100) of cameras/images to remove
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (filtered_cameras, filtered_images, filtered_points3D)
    """
    if removal_percentage <= 0:
        return cameras, images, points3D
    
    if removal_percentage >= 100:
        logging.warning("Cannot remove 100% of cameras, skipping removal")
        return cameras, images, points3D
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create deep copies to avoid modifying original data
    filtered_cameras = copy.deepcopy(cameras)
    filtered_images = copy.deepcopy(images)
    filtered_points3D = copy.deepcopy(points3D)
    
    # Get list of image IDs and randomly select ones to remove (matching reference script logic)
    all_image_ids = list(filtered_images.keys())
    random.shuffle(all_image_ids)
    
    num_to_delete = int(len(all_image_ids) * removal_percentage / 100.0)
    
    if num_to_delete <= 0:
        logging.info(f"No cameras to remove (percentage too small for {len(all_image_ids)} images)")
        return filtered_cameras, filtered_images, filtered_points3D
    
    # Take first N images from shuffled list
    ids_to_delete_this_level = all_image_ids[:num_to_delete]
    ids_to_delete_set = set(ids_to_delete_this_level)
    kept_image_ids_set = set(all_image_ids) - ids_to_delete_set
    
    logging.info(f"Removing {num_to_delete}/{len(all_image_ids)} images ({removal_percentage:.1f}%): {sorted(ids_to_delete_set)}")
    
    # Remove selected images first
    current_images = {img_id: img for img_id, img in filtered_images.items() if img_id in kept_image_ids_set}
    
    # Find camera IDs that are still referenced by remaining images
    camera_ids_in_use = set(img.camera_id for img in current_images.values())
    
    # Keep only cameras that are still referenced by remaining images
    current_cameras = {cam_id: cam for cam_id, cam in filtered_cameras.items() if cam_id in camera_ids_in_use}
    
    # Aggressive point filtering
    point_ids_observed_by_deleted_images = set()
    for img_id in ids_to_delete_set:
        if img_id in filtered_images:
            original_image = filtered_images[img_id]
            for p3d_id in original_image.point3D_ids:
                if p3d_id != -1:  # Valid point ID
                    point_ids_observed_by_deleted_images.add(p3d_id)
    
    # Keep only points NOT observed by deleted images
    current_points3D = {}
    for point_id, point in filtered_points3D.items():
        if point_id not in point_ids_observed_by_deleted_images:
            # Update track to only include kept images
            new_track_image_ids = [img_id for img_id in point.image_ids if img_id in kept_image_ids_set]
            if new_track_image_ids:  # Only keep if still observed
                new_track_point2D_idxs = [p2d_idx for img_id, p2d_idx in zip(point.image_ids, point.point2D_idxs) if img_id in kept_image_ids_set]
                current_points3D[point_id] = point._replace(
                    image_ids=np.array(new_track_image_ids),
                    point2D_idxs=np.array(new_track_point2D_idxs)
                )
    
    logging.info(f"SfM sparsification results:")
    logging.info(f"  Original: {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")
    logging.info(f"  Sparsified: {len(current_cameras)} cameras, {len(current_images)} images, {len(current_points3D)} points")
    logging.info(f"  Removed: {len(cameras) - len(current_cameras)} cameras, {len(images) - len(current_images)} images, {len(points3D) - len(current_points3D)} points")
    
    return current_cameras, current_images, current_points3D

def filter_points_by_error(points3D: dict, error_threshold: float = 0.5):
    """Filter 3D points by reprojection error."""
    filtered_points = []
    filtered_colors = []
    
    for point_id, pt in points3D.items():
        if pt.error < error_threshold:
            filtered_points.append(pt.xyz)
            filtered_colors.append(pt.rgb)
    
    if not filtered_points:
        raise ValueError(f"No points remain after filtering with error threshold {error_threshold}")
    
    points_array = np.array(filtered_points, dtype=np.float32)
    colors_array = np.array(filtered_colors, dtype=np.uint8)
    
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
    
    return cropped_points, cropped_colors

def prepare_features(points: np.ndarray, colors: np.ndarray, rotmats: np.ndarray, camera_centers: np.ndarray):
    """Prepare features for model input."""
    # Point cloud features: [x, y, z, r, g, b] (normalize colors to [0,1])
    pc_features = np.hstack([
        points.astype(np.float32),
        colors.astype(np.float32) / 255.0
    ])
    
    # Convert rotation matrices to quaternions (x, y, z, w format - scalar last)
    quats_list = []
    for R_mat in rotmats:
        try:
            quat = R.from_matrix(R_mat).as_quat(canonical=True, scalar_first=False)  # [x, y, z, w]
            quats_list.append(quat)
        except ValueError:
            logging.warning("Invalid rotation matrix, using identity quaternion")
            quats_list.append(np.array([0.0, 0.0, 0.0, 1.0]))  # Identity quaternion [x, y, z, w]
    
    camera_quats = np.array(quats_list, dtype=np.float32)
    
    # Camera pose features: [cx, cy, cz, qx, qy, qz, qw]
    pose_features = np.hstack([
        camera_centers.astype(np.float32),
        camera_quats
    ])
    
    return pc_features, pose_features

def create_batch_for_inference(pc_features: np.ndarray, pose_features: np.ndarray, device: torch.device, model_dtype=None):
    """Create a single-sample batch for model inference."""
    # Convert to tensors and add batch dimension
    pc_tensor = torch.from_numpy(pc_features).unsqueeze(0)
    pose_tensor = torch.from_numpy(pose_features).unsqueeze(0)
    
    # Apply model dtype if specified
    if model_dtype is not None:
        pc_tensor = pc_tensor.to(device=device, dtype=model_dtype)
        pose_tensor = pose_tensor.to(device=device, dtype=model_dtype)
    else:
        pc_tensor = pc_tensor.to(device)
        pose_tensor = pose_tensor.to(device)
    
    num_points = pc_features.shape[0]
    num_cameras = pose_features.shape[0]
    
    # Create cu_seqlens for FlashAttention
    pc_lengths_tensor = torch.tensor([num_points], dtype=torch.int32, device=device)
    pose_lengths_tensor = torch.tensor([num_cameras], dtype=torch.int32, device=device)
    
    pc_cu_seqlens = F.pad(torch.cumsum(pc_lengths_tensor, dim=0, dtype=torch.int32), (1, 0), value=0)
    pose_cu_seqlens = F.pad(torch.cumsum(pose_lengths_tensor, dim=0, dtype=torch.int32), (1, 0), value=0)
    
    batch = {
        'pc_input': pc_tensor,
        'pose_input': pose_tensor,
        'pc_cu_seqlens': pc_cu_seqlens,
        'pc_max_len': num_points,
        'pose_cu_seqlens': pose_cu_seqlens,
        'pose_max_len': num_cameras
    }
    
    return batch

def load_model(checkpoint_path: str, num_classes: int, device: torch.device, model_dtype):
    """Load the trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = TwoHeadTransformer(num_classes=num_classes, **DEFAULT_MODEL_PARAMS)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle DataParallel checkpoints
        if any(key.startswith('module.') for key in state_dict.keys()):
            logging.info("Detected 'module.' prefix in checkpoint keys, removing it.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=model_dtype)
        model.eval()
        
        logging.info(f"Loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise

def create_pose_error_grid(pose_errors: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 6x18 grids of pose errors from the pose_errors dictionary.
    
    Args:
        pose_errors: Dictionary with pose error data
        
    Returns:
        Tuple of (translation_error_grid, rotation_error_grid) both shape (6, 18)
    """
    # Initialize grids with NaN
    trans_error_grid = np.full((6, 18), np.nan)
    rot_error_grid = np.full((6, 18), np.nan)
    
    # Angle mappings
    x_angles = np.arange(-60, 60, 20)  # [-60, -40, -20, 0, 20, 40] (6 values)
    y_angles = np.arange(-180, 180, 20)  # [-180, -160, ..., 160] (18 values)
    
    # Create angle to index mappings
    x_angle_to_idx = {angle: i for i, angle in enumerate(x_angles)}
    y_angle_to_idx = {angle: i for i, angle in enumerate(y_angles)}
    
    for image_name, error_data in pose_errors.items():
        # Parse angles from image name (e.g., "6x18_x-60_y-180_000001.jpg")
        try:
            parts = image_name.split('_')
            # Parse x_angle: parts[1] is like "x-60" or "x20", int(parts[1][1:]) handles both cases
            x_angle = int(parts[1][1:])  # Remove 'x' prefix
            y_angle = int(parts[2][1:])  # Remove 'y' prefix
            
            # Get grid indices
            if x_angle in x_angle_to_idx and y_angle in y_angle_to_idx:
                row = x_angle_to_idx[x_angle]
                col = y_angle_to_idx[y_angle]
                
                trans_error_grid[row, col] = error_data['translation_error']
                rot_error_grid[row, col] = error_data['rotation_error']
            else:
                logging.warning(f"Angle ({x_angle}, {y_angle}) from {image_name} not in expected ranges")
                
        except (IndexError, ValueError) as e:
            logging.warning(f"Failed to parse angles from {image_name}: {e}")
            continue
    
    return trans_error_grid, rot_error_grid

def find_best_viewing_direction_with_errors(model_outputs, trans_error_grid, rot_error_grid):
    """
    Find the viewing direction with highest probability for class 0 and get corresponding GT errors.
    
    Args:
        model_outputs: Raw model outputs (logits) with shape [1, num_classes, 6, 18]
        trans_error_grid: Translation errors grid [6, 18]
        rot_error_grid: Rotation errors grid [6, 18]
        
    Returns:
        dict: Contains best direction info including GT errors at that location
    """
    # Convert logits to probabilities
    probabilities = torch.softmax(model_outputs, dim=1)  # [1, num_classes, 6, 18]
    
    # Extract class 0 probabilities
    class_0_probs = probabilities[0, 0].float().cpu().numpy()  # [6, 18]
    
    # Find the cell with maximum class 0 probability
    max_row, max_col = np.unravel_index(np.argmax(class_0_probs), class_0_probs.shape)
    max_probability = class_0_probs[max_row, max_col]
    
    # Get GT errors at the best location
    gt_trans_error = trans_error_grid[max_row, max_col]
    gt_rot_error = rot_error_grid[max_row, max_col]
    
    # Map back to rotation angles
    x_angles = np.arange(-60, 60, 20)
    y_angles = np.arange(-180, 180, 20)
    
    best_x_angle = x_angles[max_row]
    best_y_angle = y_angles[max_col]
    
    result = {
        'best_cell': (max_row, max_col),
        'best_x_angle': best_x_angle,
        'best_y_angle': best_y_angle,
        'class_0_probability': max_probability,
        'gt_translation_error': gt_trans_error,
        'gt_rotation_error': gt_rot_error,
        'class_0_prob_grid': class_0_probs
    }
    
    return result

def process_waypoint(waypoint_idx: int, waypoint_coords: np.ndarray, 
                    filtered_points: np.ndarray, filtered_colors: np.ndarray,
                    images: dict, waypoint_data: dict, model, device: torch.device, 
                    model_dtype, amp_enabled: bool, use_bf16: bool):
    """Process a single waypoint and return evaluation results."""
    
    try:
        # Transform data relative to waypoint
        transformed_points, transformed_colors, rotmats, camera_centers = transform_data(
            filtered_points, filtered_colors, images, waypoint_coords
        )
        
        # Crop to bounding box
        cropped_points, cropped_colors = crop_to_bounding_box(
            transformed_points, transformed_colors
        )
        
        # Prepare features
        pc_features, pose_features = prepare_features(
            cropped_points, cropped_colors, rotmats, camera_centers
        )
        
        # Create batch
        batch = create_batch_for_inference(pc_features, pose_features, device, model_dtype)
        
        # Run inference
        with torch.no_grad():
            with torch.autocast(device_type=device.type, 
                              dtype=torch.bfloat16 if use_bf16 else torch.float16, 
                              enabled=amp_enabled):
                outputs = model(**batch)
        
        # Create error grids from waypoint data
        pose_errors = waypoint_data.get('pose_errors', {})
        
        if not pose_errors:
            logging.warning(f"No pose errors found for waypoint {waypoint_idx + 1}")
            return None
        
        trans_error_grid, rot_error_grid = create_pose_error_grid(pose_errors)
        
        # Find best viewing direction and get GT errors
        best_direction = find_best_viewing_direction_with_errors(
            outputs, trans_error_grid, rot_error_grid
        )
        
        return best_direction
        
    except Exception as e:
        logging.error(f"Failed to process waypoint {waypoint_idx + 1}: {e}", exc_info=True)
        return None

def evaluate_scene(data_parser: ActLocDataParser, scene_name: str, model, device: torch.device,
                  model_dtype, amp_enabled: bool, use_bf16: bool, error_threshold: float = 0.5,
                  removal_percentage: float = 0.0):
    """Evaluate a single scene and return results for each waypoint."""
    
    logging.info(f"Evaluating scene: {scene_name} (sparsifying {removal_percentage:.1f}% of cameras)")
    
    try:
        # Load scene data
        scene_data = data_parser.load_scene_data(scene_name)
        
        waypoints = scene_data['waypoints']
        cameras = scene_data['cameras']
        images = scene_data['images']
        points3D = scene_data['points3D']
        waypoint_data = scene_data['waypoint_data']
        
        # Apply SfM sparsification if requested
        if removal_percentage > 0:
            cameras, images, points3D = remove_cameras_and_associated_points(
                cameras, images, points3D, removal_percentage, seed=42
            )
            
            # Check if we still have enough data
            if len(images) == 0:
                logging.error(f"No images remain after sparsifying {removal_percentage}% for scene {scene_name}")
                return None
            if len(points3D) == 0:
                logging.error(f"No 3D points remain after sparsifying {removal_percentage}% for scene {scene_name}")
                return None
        
        # Filter points by error
        filtered_points, filtered_colors = filter_points_by_error(points3D, error_threshold)
        
        scene_results = []
        
        # Process each waypoint
        for waypoint_idx, waypoint_coords in enumerate(waypoints):
            waypoint_num = waypoint_idx + 1  # 1-based indexing
            
            if waypoint_num not in waypoint_data:
                logging.warning(f"No localization data found for waypoint {waypoint_num} in scene {scene_name}")
                scene_results.append(None)
                continue
            
            wp_data = waypoint_data[waypoint_num]
            result = process_waypoint(
                waypoint_idx, waypoint_coords, filtered_points, filtered_colors,
                images, wp_data, model, device, model_dtype, amp_enabled, use_bf16
            )
            
            scene_results.append(result)
        
        return scene_results
        
    except Exception as e:
        logging.error(f"Failed to evaluate scene {scene_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate ActLoc model on new data structure with SfM sparsification")
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='full_actloc_data/test_data',
                       help='Root directory of ActLoc dataset')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                       help='Specific scenes to evaluate (if not specified, evaluates all)')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/actloc_binary_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--num-classes', type=int, choices=[2, 4], default=2,
                       help='Number of output classes')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run evaluation on')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--error-threshold', type=float, default=0.5,
                       help='Reprojection error threshold for point filtering')
    
    # SfM sparsification arguments
    parser.add_argument('--enable-sparsification', action='store_true', default=False,
                       help='Enable SfM sparsification study (removing cameras and associated points)')
    parser.add_argument('--sparsification-percentages', type=float, nargs='+', 
                       default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                       help='Percentages of cameras to remove for sparsification study')
    
    # Output arguments
    parser.add_argument('--output-file', type=str, default='./evaluation_results.txt',
                       help='Output file for detailed results')
    parser.add_argument('--sparsification-output-file', type=str, default='./sparsification_results.txt',
                       help='Output file for sparsification study results')
    
    args = parser.parse_args()
    
    # Setup device and logging
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Setup AMP
    use_bf16 = False
    amp_enabled = args.amp and device.type == 'cuda'
    if amp_enabled:
        use_bf16 = torch.cuda.is_bf16_supported()
        logging.info(f"AMP enabled. Using {'bfloat16' if use_bf16 else 'float16'}.")
    
    # Set model dtype
    model_dtype = torch.float32
    if amp_enabled:
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    try:
        # Initialize data parser
        data_parser = ActLocDataParser(args.data_root)
        
        # Get scenes to evaluate
        if args.scenes is None:
            scenes_to_evaluate = data_parser.list_scenes()
            logging.info(f"Evaluating all {len(scenes_to_evaluate)} scenes")
        else:
            scenes_to_evaluate = args.scenes
            logging.info(f"Evaluating {len(scenes_to_evaluate)} specified scenes")
        
        # Load model
        model = load_model(args.checkpoint, args.num_classes, device, model_dtype)
        
        # Define thresholds (translation in meters, rotation in degrees)
        thresholds = [(0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (5.0, 10.0)]
        
        if args.enable_sparsification:
            # Run sparsification study
            logging.info("=== Running SfM Sparsification Study ===")
            sparsification_results = {}
            
            for removal_percentage in args.sparsification_percentages:
                logging.info(f"\n--- Evaluating with {removal_percentage}% camera removal ---")
                
                waypoint_counts_below_threshold = [0] * len(thresholds)
                total_waypoints_processed = 0
                all_results = {}
                
                # Evaluate each scene with current removal percentage
                for scene_name in scenes_to_evaluate:
                    scene_results = evaluate_scene(
                        data_parser, scene_name, model, device, model_dtype, 
                        amp_enabled, use_bf16, args.error_threshold, removal_percentage
                    )
                    
                    if scene_results is not None:
                        all_results[scene_name] = scene_results
                        
                        # Process results for threshold evaluation
                        for waypoint_idx, result in enumerate(scene_results):
                            if result is not None:
                                gt_trans_error = result['gt_translation_error']
                                gt_rot_error = result['gt_rotation_error']
                                
                                # Check if GT errors are valid
                                if not np.isnan(gt_trans_error) and not np.isnan(gt_rot_error):
                                    total_waypoints_processed += 1
                                    
                                    # Check against each threshold
                                    for j, (t_thresh, r_thresh) in enumerate(thresholds):
                                        if gt_trans_error <= t_thresh and gt_rot_error <= r_thresh:
                                            waypoint_counts_below_threshold[j] += 1
                    else:
                        logging.error(f"Failed to evaluate scene {scene_name} with {removal_percentage}% sparsification")
                
                # Store results for this removal percentage
                sparsification_results[removal_percentage] = {
                    'total_waypoints': total_waypoints_processed,
                    'threshold_counts': waypoint_counts_below_threshold.copy(),
                    'all_results': all_results
                }
                
                # Print results for current removal percentage
                logging.info(f"Results for {removal_percentage}% sparsification: {total_waypoints_processed} waypoints")
                if total_waypoints_processed > 0:
                    for i, (t_thresh, r_thresh) in enumerate(thresholds):
                        count = waypoint_counts_below_threshold[i]
                        percentage = (count / total_waypoints_processed) * 100
                        logging.info(f"  ≤ {t_thresh:.2f}m / {r_thresh:.1f}°: {count}/{total_waypoints_processed} ({percentage:.2f}%)")
            
            # Save sparsification results
            with open(args.sparsification_output_file, 'w') as f:
                f.write("ActLoc Model SfM Sparsification Study Results\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("Summary Table:\n")
                f.write("Sparsification% | Total WPs | ≤0.1m/1° | ≤0.25m/2° | ≤0.5m/5° | ≤5m/10°\n")
                f.write("-" * 70 + "\n")
                
                for removal_percentage in args.sparsification_percentages:
                    results = sparsification_results[removal_percentage]
                    total = results['total_waypoints']
                    counts = results['threshold_counts']
                    
                    if total > 0:
                        percentages = [f"{(count/total)*100:.1f}%" for count in counts]
                        f.write(f"{removal_percentage:12.1f}% | {total:8d} | {percentages[0]:7s} | {percentages[1]:8s} | {percentages[2]:7s} | {percentages[3]:7s}\n")
                    else:
                        f.write(f"{removal_percentage:12.1f}% | {total:8d} | {'N/A':7s} | {'N/A':8s} | {'N/A':7s} | {'N/A':7s}\n")
                
                f.write("\nDetailed Results:\n")
                f.write("-" * 30 + "\n")
                
                for removal_percentage in args.sparsification_percentages:
                    results = sparsification_results[removal_percentage]
                    f.write(f"\n{removal_percentage}% Camera Sparsification:\n")
                    f.write(f"Total waypoints evaluated: {results['total_waypoints']}\n")
                    
                    if results['total_waypoints'] > 0:
                        f.write("Threshold Performance:\n")
                        for i, (t_thresh, r_thresh) in enumerate(thresholds):
                            count = results['threshold_counts'][i]
                            percentage = (count / results['total_waypoints']) * 100
                            f.write(f"  ≤ {t_thresh:.2f}m / {r_thresh:.1f}°: {count}/{results['total_waypoints']} waypoints ({percentage:.2f}%)\n")
            
            logging.info(f"Sparsification study results saved to {args.sparsification_output_file}")
            
        else:
            # Run standard evaluation (0% removal)
            logging.info("=== Running Standard Evaluation ===")
            
            waypoint_counts_below_threshold = [0] * len(thresholds)
            total_waypoints_processed = 0
            all_results = {}
            
            # Evaluate each scene
            for scene_name in scenes_to_evaluate:
                scene_results = evaluate_scene(
                    data_parser, scene_name, model, device, model_dtype, 
                    amp_enabled, use_bf16, args.error_threshold, removal_percentage=0.0
                )
                
                if scene_results is not None:
                    all_results[scene_name] = scene_results
                    
                    # Process results for threshold evaluation
                    for waypoint_idx, result in enumerate(scene_results):
                        if result is not None:
                            gt_trans_error = result['gt_translation_error']
                            gt_rot_error = result['gt_rotation_error']
                            
                            # Check if GT errors are valid
                            if not np.isnan(gt_trans_error) and not np.isnan(gt_rot_error):
                                total_waypoints_processed += 1
                                
                                # Check against each threshold
                                for j, (t_thresh, r_thresh) in enumerate(thresholds):
                                    if gt_trans_error <= t_thresh and gt_rot_error <= r_thresh:
                                        waypoint_counts_below_threshold[j] += 1
                                
                                # Log individual results
                                logging.info(f"{scene_name}/waypoint_{waypoint_idx + 1}: "
                                           f"Best view at ({result['best_x_angle']}°, {result['best_y_angle']}°), "
                                           f"GT errors: {gt_trans_error:.4f}m / {gt_rot_error:.4f}°, "
                                           f"Class 0 prob: {result['class_0_probability']:.4f}")
                else:
                    logging.error(f"Failed to evaluate scene {scene_name}")
            
            # Print final results
            logging.info("\n=== EVALUATION RESULTS ===")
            logging.info(f"Total waypoints evaluated: {total_waypoints_processed}")
            
            if total_waypoints_processed > 0:
                logging.info("\nThreshold Performance:")
                for i, (t_thresh, r_thresh) in enumerate(thresholds):
                    count = waypoint_counts_below_threshold[i]
                    percentage = (count / total_waypoints_processed) * 100
                    logging.info(f"  ≤ {t_thresh:.2f}m / {r_thresh:.1f}°: {count}/{total_waypoints_processed} waypoints ({percentage:.2f}%)")
                
                # Save detailed results
                with open(args.output_file, 'w') as f:
                    f.write("ActLoc Model Evaluation Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Total waypoints evaluated: {total_waypoints_processed}\n\n")
                    f.write("Threshold Performance:\n")
                    for i, (t_thresh, r_thresh) in enumerate(thresholds):
                        count = waypoint_counts_below_threshold[i]
                        percentage = (count / total_waypoints_processed) * 100
                        f.write(f"  ≤ {t_thresh:.2f}m / {r_thresh:.1f}°: {count}/{total_waypoints_processed} waypoints ({percentage:.2f}%)\n")
                    
                    f.write("\nDetailed Results by Scene:\n")
                    f.write("-" * 30 + "\n")
                    for scene_name, scene_results in all_results.items():
                        f.write(f"\nScene: {scene_name}\n")
                        for waypoint_idx, result in enumerate(scene_results):
                            if result is not None:
                                f.write(f"  Waypoint {waypoint_idx + 1}: "
                                      f"Best view ({result['best_x_angle']}°, {result['best_y_angle']}°), "
                                      f"GT errors {result['gt_translation_error']:.4f}m / {result['gt_rotation_error']:.4f}°, "
                                      f"Prob {result['class_0_probability']:.4f}\n")
                            else:
                                f.write(f"  Waypoint {waypoint_idx + 1}: Failed to process\n")
                
                logging.info(f"Detailed results saved to {args.output_file}")
            else:
                logging.warning("No waypoints were successfully evaluated")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()