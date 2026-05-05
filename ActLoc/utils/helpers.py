import logging
import numpy as np
import torch

from ActLoc.actloc_core.processing import transform_data, crop_to_bounding_box, prepare_features_for_one_waypoint
from ActLoc.actloc_core.torch_utils import create_single_sample_batch_for_inference



def find_best_viewing_direction_for_one_waypoint(model_outputs, waypoint_coords):
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
    # X-axis (elevation): 6 cells covering [-60, 60) with interval 20
    # Y-axis (azimuth): 18 cells covering [-180, 180) with interval 20
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

def run_inference_for_one_waypoint(waypoint: np.ndarray, waypoint_idx: int, 
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
        pc_features, pose_features = prepare_features_for_one_waypoint(
            cropped_points, cropped_colors, rotmats, camera_centers
        )
        
        # Create batch
        batch = create_single_sample_batch_for_inference(pc_features, pose_features, device, model_dtype)
        
        # Run inference
        with torch.no_grad():
            with torch.autocast(device_type=device.type, 
                              dtype=torch.bfloat16 if use_bf16 else torch.float16, 
                              enabled=amp_enabled):
                outputs = model(**batch)
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        output_probs = torch.softmax(outputs, dim=1)
        good_vp_probs = output_probs[0, 0].unsqueeze(0) # [0,1] -> [good, bad]
        good_vp_probs = good_vp_probs.float().cpu().numpy()  # [1, 6, 18]
        prediction_grid = predictions[0].cpu().numpy()  # [6, 18]
        
        # Find best viewing direction for class 0
        best_direction = find_best_viewing_direction_for_one_waypoint(outputs, waypoint)

        return prediction_grid, good_vp_probs, best_direction

    except Exception as e:
        logging.error(f"Failed to process waypoint {waypoint_idx}: {e}", exc_info=True)
        return None