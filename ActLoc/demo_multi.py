import argparse
import logging
import sys
import numpy as np
import torch
from actloc_core.io import load_sfm_model
from actloc_core.processing import filter_points_by_error, calculate_extrinsic_matrix
from actloc_core.torch_utils import load_model
from utils.helpers import run_inference_for_one_waypoint
from utils.io import load_waypoints
from planning import hybrid_cost_vp_selection,vps_to_extrinsic

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run inference on SfM scene with waypoints")
    
    parser.add_argument('--sfm-dir', type=str, default='./example_data/00005_reference_sfm',
                       help='Path to COLMAP SfM reconstruction folder')
    parser.add_argument('--waypoints-file', type=str, default='./example_data/waypoints.txt',
                       help='Path to text file containing waypoint coordinates')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/trained_actloc.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-file', type=str, default='./example_data/predicted_poses.npz',
                       help='Output file for predicted poses (.npz format)')
    parser.add_argument('--lamda', type=float, default=0.02,
                       help='Lambda parameter for balancing motion and localization, higher values prioritize motion, 0-0.5')
    args = parser.parse_args()

    # Setup device and logging
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This model requires a NVIDIA GPU with CUDA support to run.")
        sys.exit(1)

    device = torch.device('cuda')
    logging.info(f"Using device: {device}")
    
    # Setup AMP
    amp_enabled = True
    use_bf16 = torch.cuda.is_bf16_supported()
    logging.info(f"AMP is enabled. Using {'bfloat16' if use_bf16 else 'float16'}.")
    
    try:
        # Load data
        logging.info("Loading SfM model and waypoints...")
        _, images, points3D = load_sfm_model(args.sfm_dir)
        waypoints = load_waypoints(args.waypoints_file)
        
        # Filter points by error
        filtered_points, filtered_colors = filter_points_by_error(points3D)
        
        # Load model
        model = load_model(args.checkpoint, device)
        
        # Set model dtype for AMP
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
        model.to(dtype=model_dtype)
        logging.info(f"Model set to dtype: {model_dtype}")
        
        # Process each waypoint
        print("\n--- Inference Results ---")
        single_results = []
        for waypoint_idx, waypoint in enumerate(waypoints):
            result = run_inference_for_one_waypoint(
                waypoint, waypoint_idx, filtered_points, filtered_colors,
                images, model, device, amp_enabled, use_bf16
            )
            single_results.append(result)
        
        pred_map = []
        for item in single_results:
            pred_map.append(item[1])  
        vps, probs = hybrid_cost_vp_selection(np.array(pred_map).squeeze(), args.lamda)
        cam_extrinsic = vps_to_extrinsic(np.array(vps), waypoints).squeeze()

        np.savez(
            args.output_file,
            waypoints=waypoints,
            extrinsics=np.array(cam_extrinsic),
            angles=np.array(vps), # TODO: placeholder, need to convert to angles
            probabilities=np.array(probs)
        )
        logging.info(f"Saved predicted poses to {args.output_file}")
        
        logging.info("Inference completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
