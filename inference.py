import argparse
import logging
import os
import sys
import numpy as np

try:
    from utils.read_write_model import read_model
except ImportError as e:
    logging.error(f"Failed to import required local modules: {e}")
    logging.error("ensure utils/read_write_model.py is accessible")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_waypoints(waypoints_file: str) -> np.ndarray:
    """load waypoint coordinates from text file"""
    if not os.path.exists(waypoints_file):
        raise FileNotFoundError(f"waypoints file not found: {waypoints_file}")
    
    waypoints = np.loadtxt(waypoints_file, dtype=np.float32)
    if waypoints.ndim == 1 and waypoints.shape[0] == 3:
        waypoints = waypoints.reshape(1, 3)
    elif waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"expected waypoints shape (N, 3), got {waypoints.shape}")
    
    logging.info(f"loaded {waypoints.shape[0]} waypoints from {waypoints_file}")
    return waypoints

def load_sfm_model(sfm_dir: str):
    """load colmap sfm reconstruction model"""
    if not os.path.isdir(sfm_dir):
        raise FileNotFoundError(f"sfm directory not found: {sfm_dir}")
    
    # try binary format first, fallback to text
    try:
        cameras, images, points3D = read_model(sfm_dir, ext=".bin")
        logging.info(f"loaded sfm model (binary): {len(images)} images, {len(points3D)} points")
    except Exception:
        try:
            cameras, images, points3D = read_model(sfm_dir, ext=".txt")
            logging.info(f"loaded sfm model (text): {len(images)} images, {len(points3D)} points")
        except Exception as e:
            raise RuntimeError(f"failed to load sfm model from {sfm_dir}: {e}")
    
    if not images:
        raise ValueError("no images found in the sfm model")
    
    return cameras, images, points3D

def filter_points_by_error(points3D: dict, error_threshold: float = 0.5):
    """filter 3d points by reprojection error"""
    filtered_points = []
    
    for point_id, pt in points3D.items():
        if pt.error < error_threshold:
            filtered_points.append(pt.xyz)
    
    if not filtered_points:
        raise ValueError(f"no points remain after filtering with error threshold {error_threshold}")
    
    points_array = np.array(filtered_points, dtype=np.float32)
    
    logging.info(f"filtered points: kept {len(points_array)} out of {len(points3D)} points (error < {error_threshold})")
    return points_array

# important: you can modify the signature of this function if needed,
# just make sure the input remains waypoints: np.ndarray
def predict_best_angles_per_pose(waypoints: np.ndarray, points):
    # the sample method is based on maximizing visibility
    from method.max_visibility import predict_pose

    print("\n--- inference results ---")
    best_angles = []  # store best angles for each waypoint
    filtered_points = filter_points_by_error(points)  # removes noisy points

    for waypoint_idx, waypoint in enumerate(waypoints):
        best_elev, best_az = predict_pose(waypoint, waypoint_idx, filtered_points)
        best_angles.append(np.array((best_elev, best_az)))
    
    return best_angles

def main():
    parser = argparse.ArgumentParser(description="run inference on sfm scene with waypoints")
    
    # required arguments
    parser.add_argument('--sfm-dir', type=str, default='./example_data/00005_reference_sfm',
                        help='path to colmap sfm reconstruction folder')
    parser.add_argument('--waypoints-file', type=str, default='./example_data/sampled_waypoints.txt',
                        help='path to text file containing waypoint coordinates [required]', required=True)
    parser.add_argument('--output-angles', type=str, default='./example_data/best_viewing_angles.txt',
                        help='output file to save best viewing angles for each waypoint [required]', required=True)
    args = parser.parse_args()

    try:
        # load data
        logging.info("loading sfm model and waypoints...")
        # note: only 3d points are used here, but images and cameras are available if needed
        cameras, images, points3D = load_sfm_model(args.sfm_dir)
        waypoints = load_waypoints(args.waypoints_file)

        best_angles = predict_best_angles_per_pose(waypoints, points3D)

        # save best angles to text file
        if best_angles:
            best_angles_array = np.array(best_angles)
            np.savetxt(args.output_angles, best_angles_array, fmt='%g', 
                       header='X_angle Y_angle (degrees)', comments='# ')
            logging.info(f"saved best viewing angles to {args.output_angles}")
        
        logging.info("inference completed successfully!")
        
    except Exception as e:
        logging.error(f"inference failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
