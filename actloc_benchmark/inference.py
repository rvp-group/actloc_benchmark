import argparse
import logging
import os
import sys
from scipy.spatial.transform import Rotation as R

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

try:
    from utils.io import *
except ImportError as e:
    logging.error(f"Failed to import required local modules: {e}")
    logging.error("ensure utils/io.py is accessible")
    sys.exit(1)


from method.your_method import actloc_predict_best_angles



def predict_best_angles_per_pose(input: dict):

    best_angles = {}  # store best angles for each waypoint


    checkpoint_path = "ActLoc/checkpoints/trained_actloc.pth"
    best_angles = actloc_predict_best_angles(input, checkpoint_path)
    ## Make Changes Above This Line
    return best_angles





def main():
    parser = argparse.ArgumentParser(
        description="run inference on sfm scene with waypoints"
    )

    parser.add_argument(
        "--sfm-dir",
        type=str,
        default="./example_data/00010-DBjEcHFg4oq/scene_reconstruction",
        help="path to colmap sfm reconstruction folder",
    )
    parser.add_argument(
        "--waypoints-file",
        type=str,
        default="./example_data/00010-DBjEcHFg4oq/sampled_waypoints.txt",
        help="path to text file containing waypoint coordinates [required]",
        required=True,
    )
    parser.add_argument(
        "--output-estimate",
        type=str,
        default="./example_data/estimate/pose_estimate.txt",
        help="output file to save best viewing angles for each waypoint [required]",
        required=True,
    )
    args = parser.parse_args()

    try:
        # load data
        logging.info("loading sfm model and waypoints...")
        cameras, images, points3D = load_sfm_model(args.sfm_dir)
        waypoints = load_waypoints(args.waypoints_file)

        input = {
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "waypoints": waypoints,
        }

        best_angles = predict_best_angles_per_pose(input)
        # save best angles to text file
        if best_angles:
            assert len(best_angles.keys()) == len(waypoints.keys())
            output_dir = os.path.dirname(args.output_estimate)
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"writing results in: {output_dir}")

            write_colmap_pose_file(waypoints, best_angles, args.output_estimate)
            logging.info(f"saved best viewing angles to {args.output_estimate}")

        logging.info("inference completed successfully!")

    except Exception as e:
        logging.error(f"inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
