import argparse
import logging
import os
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

try:
    from utils.io import *
except ImportError as e:
    logging.error(f"Failed to import required local modules: {e}")
    logging.error("ensure utils/io.py is accessible")
    sys.exit(1)

# IMPORT YOUR METHOD BELOW
# from method.your_method import your_method_function  # replace with your actual method function

# this is one provided example
from method.max_visibility import predict_pose, filter_points_by_error

############


## IMPORTANT:
# This is the wrapper function that will be called by the benchmark script.
# You should not change the function signature or the input/output format.
# Instead, implement your method logic inside the predict_best_angles_per_pose function.


def predict_best_angles_per_pose(input: dict):

    best_angles = []  # store best angles for each waypoint

    ## Make Changes Below This Line
    filtered_points = filter_points_by_error(input["points3D"])
    for waypoint_idx, waypoint in enumerate(input["waypoints"]):
        best_elev, best_az = predict_pose(waypoint, waypoint_idx, filtered_points)
        best_angles.append(np.array((best_elev, best_az)))
    ## Make Changes Above This Line

    return best_angles


def main():
    parser = argparse.ArgumentParser(
        description="run inference on sfm scene with waypoints"
    )

    # required arguments
    parser.add_argument(
        "--sfm-dir",
        type=str,
        default="./example_data/00005_reference_sfm",
        help="path to colmap sfm reconstruction folder",
    )
    parser.add_argument(
        "--waypoints-file",
        type=str,
        default="./example_data/sampled_waypoints.txt",
        help="path to text file containing waypoint coordinates [required]",
        required=True,
    )
    parser.add_argument(
        "--output-angles",
        type=str,
        default="./example_data/best_viewing_angles.txt",
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
            assert len(best_angles) == len(waypoints)

            best_angles_array = np.array(best_angles)
            np.savetxt(
                args.output_angles,
                best_angles_array,
                fmt="%g",
                header="X_angle Y_angle (degrees)",
                comments="# ",
            )

            colmap_estimate_fn = os.path.splitext(args.output_angles)[0] + "_colmap.txt"
            write_colmap_pose_file(waypoints, best_angles_array, colmap_estimate_fn)
            logging.info(
                f"saved best viewing angles to {args.output_angles} and COLMAP style estimate to {colmap_estimate_fn}"
            )

        logging.info("inference completed successfully!")

    except Exception as e:
        logging.error(f"inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
