import os
import logging
from ActLoc.utils.read_write_model import read_model

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