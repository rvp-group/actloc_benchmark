import argparse
import logging
import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from utils.read_write_model import read_model, qvec2rotmat
    from models.new_model import TwoHeadTransformer
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Ensure utils/read_write_model.py and models/new_model.py are accessible")
    sys.exit(1)

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

def load_sfm_model(sfm_dir: str):
    """Load COLMAP SfM reconstruction model."""
    if not os.path.isdir(sfm_dir):
        raise FileNotFoundError(f"SfM directory not found: {sfm_dir}")
    
    # Try binary format first, fallback to text
    try:
        cameras, images, points3D = read_model(sfm_dir, ext=".bin")
        logging.info(f"Loaded SfM model (binary): {len(images)} images, {len(points3D)} points")
    except Exception:
        try:
            cameras, images, points3D = read_model(sfm_dir, ext=".txt")
            logging.info(f"Loaded SfM model (text): {len(images)} images, {len(points3D)} points")
        except Exception as e:
            raise RuntimeError(f"Failed to load SfM model from {sfm_dir}: {e}")
    
    if not images:
        raise ValueError("No images found in the SfM model")
    
    return cameras, images, points3D

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
    
    logging.info(f"Prepared features - PC: {pc_features.shape}, Pose: {pose_features.shape}")
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

def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load the trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = TwoHeadTransformer(num_classes=num_classes, **DEFAULT_MODEL_PARAMS)
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {checkpoint_path}")
    return model

def find_best_viewing_direction(model_outputs, waypoint_coords):
    """
    Find the viewing direction with highest probability for class 0 (good accuracy).
    
    Args:
        model_outputs: Raw model outputs (logits) with shape [1, num_classes, 6, 18]
        waypoint_coords: The waypoint coordinates for reference
        
    Returns:
        dict: Contains best direction info including angles and probabilities
    """
    # Convert logits to probabilities
    probabilities = torch.softmax(model_outputs, dim=1)  # [1, num_classes, 6, 18]
    
    # Extract class 0 probabilities (good accuracy class) and convert to float32 for numpy
    class_0_probs = probabilities[0, 0].float().cpu().numpy()  # [6, 18]
    
    # Find the cell with maximum class 0 probability
    max_row, max_col = np.unravel_index(np.argmax(class_0_probs), class_0_probs.shape)
    max_probability = class_0_probs[max_row, max_col]
    
    # Map back to rotation angles
    # X-axis (elevation): 6 cells covering [-60, 40] with interval 20
    # Y-axis (azimuth): 18 cells covering [-180, 160] with interval 20
    x_angles = np.arange(-60, 60, 20)  # [-60, -40, -20, 0, 20, 40] (6 values)
    y_angles = np.arange(-180, 180, 20)  # [-180, -160, ..., 160] (18 values)
    
    best_x_angle = x_angles[max_row]
    best_y_angle = y_angles[max_col]
    
    # Get all class probabilities for the best cell and convert to float32 for numpy
    all_probs = probabilities[0, :, max_row, max_col].float().cpu().numpy()
    
    result = {
        'waypoint_coords': waypoint_coords,
        'best_cell': (max_row, max_col),
        'best_x_angle': best_x_angle,
        'best_y_angle': best_y_angle,
        'class_0_probability': max_probability,
        'all_class_probabilities': all_probs,
        'class_0_prob_grid': class_0_probs
    }
    
    return result

def run_inference_for_waypoint(waypoint: np.ndarray, waypoint_idx: int, 
                              filtered_points: np.ndarray, filtered_colors: np.ndarray,
                              images: dict, model, device: torch.device, amp_enabled: bool, use_bf16: bool):
    """Run inference for a single waypoint."""
    logging.info(f"Processing waypoint {waypoint_idx}: {waypoint}")
    
    # Get model dtype
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if amp_enabled else torch.float32
    
    try:
        # Transform data relative to waypoint
        transformed_points, transformed_colors, rotmats, camera_centers = transform_data(
            filtered_points, filtered_colors, images, waypoint
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
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        prediction_grid = predictions[0].cpu().numpy()  # [6, 18]
        
        # Find best viewing direction for class 0
        best_direction = find_best_viewing_direction(outputs, waypoint)
        
        return prediction_grid, best_direction
        
    except Exception as e:
        logging.error(f"Failed to process waypoint {waypoint_idx}: {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description="Run inference on SfM scene with waypoints")
    
    # Required arguments
    parser.add_argument('--sfm-dir', type=str, default='./example_data/00005_reference_sfm',
                       help='Path to COLMAP SfM reconstruction folder')
    parser.add_argument('--waypoints-file', type=str, default='./example_data/sampled_viewpoints.txt',
                       help='Path to text file containing waypoint coordinates')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/actloc_binary_best.pth',
                       help='Path to trained model checkpoint')
    
    # Optional arguments
    parser.add_argument('--num-classes', type=int, choices=[2, 4], default=2,
                       help='Number of output classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--output-angles', type=str, default='./example_data/best_viewing_angles.txt',
                       help='Output file to save best viewing angles for each waypoint')
    parser.add_argument('--error-threshold', type=float, default=0.5,
                       help='Reprojection error threshold for point filtering')
    
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
    
    try:
        # Load data
        logging.info("Loading SfM model and waypoints...")
        cameras, images, points3D = load_sfm_model(args.sfm_dir)
        waypoints = load_waypoints(args.waypoints_file)
        
        # Filter points by error
        filtered_points, filtered_colors = filter_points_by_error(points3D, args.error_threshold)
        
        # Load model
        model = load_model(args.checkpoint, args.num_classes, device)
        
        # Set model dtype for AMP
        if amp_enabled:
            model_dtype = torch.bfloat16 if use_bf16 else torch.float16
            model.to(dtype=model_dtype)
            logging.info(f"Model set to dtype: {model_dtype}")
        
        # Process each waypoint
        print("\n--- Inference Results ---")
        best_angles = []  # Store best angles for each waypoint
        
        for waypoint_idx, waypoint in enumerate(waypoints):
            result = run_inference_for_waypoint(
                waypoint, waypoint_idx, filtered_points, filtered_colors,
                images, model, device, amp_enabled, use_bf16
            )
            
            if result is not None:
                prediction_grid, best_direction = result
                
                # Store the best angles
                best_angles.append([best_direction['best_x_angle'], best_direction['best_y_angle']])
                
                print(f"\nWaypoint {waypoint_idx + 1} Coordinates: {waypoint}")
                print("Predicted Grid (argmax classes):")
                print(prediction_grid)
                print("\n=== Best Viewing Direction for Class 0 (Good Accuracy) ===")
                print(f"Best cell position: Row {best_direction['best_cell'][0]}, Col {best_direction['best_cell'][1]}")
                print(f"Best viewing angles: X={best_direction['best_x_angle']}°, Y={best_direction['best_y_angle']}°")
                print(f"Class 0 probability: {best_direction['class_0_probability']:.4f}")
                if len(best_direction['all_class_probabilities']) > 1:
                    print("All class probabilities at best cell:")
                    for i, prob in enumerate(best_direction['all_class_probabilities']):
                        print(f"  Class {i}: {prob:.4f}")
                print("-" * 50)
            else:
                # If processing failed, store NaN values
                best_angles.append([float('nan'), float('nan')])
                print(f"\nWaypoint {waypoint_idx + 1}: Failed to process")
                print("-" * 50)
        
        # Save best angles to text file
        if best_angles:
            best_angles_array = np.array(best_angles)
            np.savetxt(args.output_angles, best_angles_array, fmt='%g', 
                      header='X_angle Y_angle (degrees)', comments='# ')
            logging.info(f"Saved best viewing angles to {args.output_angles}")
        
        logging.info("Inference completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
