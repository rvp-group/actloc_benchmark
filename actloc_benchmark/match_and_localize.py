import argparse
import numpy as np
import pycolmap
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_exhaustive, visualization
from hloc.utils import viz_3d
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

import matplotlib.pyplot as plt
from utils.io import *

from camera import Camera

cam = Camera()  # instantiate one for all

feature_conf = extract_features.confs["superpoint_max"]
matcher_conf = match_features.confs["superpoint+lightglue"]


def numpy2rigid3d(cam_in_world):
    return pycolmap.Rigid3d(cam_in_world[:3, :])


def get_cameras(input_file):
    """
    Fixed function to get cameras, not parsed from file, value hardcoded for challenge
    see camera.py for details
    """
    input_file = Path(input_file)
    cameras = dict()

    with input_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "#":
                continue
            if len(parts) != 8:
                raise InvalidPoseLineError(
                    f"invalid number of elements in line: {line.strip()}"
                )

            img_name = parts[0] + ".png"

            camera = pycolmap.Camera(
                model=cam.model,
                width=cam.W,
                height=cam.H,
                params=[cam.fx, cam.fy, cam.cx, cam.cy],
            )
            cameras[img_name] = camera

    print(f"created {len(cameras)} cameras")
    return cameras


# class InvalidCameraLineError(Exception):
#     """Custom exception to handle invalid camera lines."""
#     pass


# def parse_camera_file(input_file):
#     """
#     Parses a text file containing image names and intrinsics to create COLMAP camera objects.

#     Args:
#         input_file (str or Path): Path to the input text file.
#             Each line should be formatted as:
#             <image_name> <model> <width> <height> <fx> <fy> <cx> <cy>
#         cameras_dict (dict): Dictionary to store image names as keys and pycolmap.Camera objects as values.

#     Returns:
#         None
#     """
#     input_file = Path(input_file)
#     cameras = dict()

#     with input_file.open("r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 raise InvalidCameraLineError(f"invalid number of elements in line: {line.strip()}")

#             img_name, model, width, height, fx, fy, cx, cy = parts

#             width, height = int(width), int(height)
#             fx, fy, cx, cy = map(float, (fx, fy, cx, cy))

#             camera = pycolmap.Camera(
#                 model=model,
#                 width=width,
#                 height=height,
#                 params=[fx, fy, cx, cy],
#             )
#             cameras[img_name] = camera

#     print(f"loaded {len(cameras)} cameras")
#     return cameras


class MismatchedBufferError(Exception):
    """Custom exception for mismatched buffer sizes or strings."""

    pass


def check_buffers(list_buffer, dict_buffer):
    """
    Check if the length of the list and dictionary are equal and that their strings match.

    Args:
        list_buffer (list of str): List of strings.
        dict_buffer (dict): Dictionary with strings as keys.

    Raises:
        MismatchedBufferError: If the lengths do not match or the strings do not match.
    """
    # check if the length of the list and dictionary are the same
    if len(list_buffer) != len(dict_buffer.keys()):
        raise MismatchedBufferError("List and dictionary have mismatched lengths.")

    # check if the strings in the list match the keys in the dictionary
    for item in list_buffer:
        if item not in dict_buffer:
            raise MismatchedBufferError(
                f"String '{item}' in the list does not match any key in the dictionary."
            )


def get_image_filenames(image_dir, image_extensions=(".png", ".jpg", ".jpeg", ".tiff")):
    """
    Get sorted list of image filenames from a directory, filtering by specified extensions.

    Args:
        image_dir: Path to the directory containing images.
        image_extensions: Tuple of image extensions to consider (default: ('.png', '.jpg', '.jpeg', '.tiff')).

    Returns:
        A tuple of two lists:
            - `queries_fn`: Relative paths of image files (filenames).
            - `queries_fullp_fn`: Full paths of image files.
    """
    image_files = sorted(
        [p for p in Path(image_dir).iterdir() if p.suffix.lower() in image_extensions]
    )

    # get relative paths and full paths for the image files
    queries_fn = [p.relative_to(image_dir).as_posix() for p in image_files]
    queries_fullp_fn = [p.as_posix() for p in image_files]

    return queries_fn, queries_fullp_fn


def main(
    ref_images_path,
    query_images_path,
    sfm_model_path,
    ref_features_fn,
    #  cameras_fn,
    poses_fn,
    output_path,
    debug=True,
):

    images = Path(ref_images_path)
    query_images = Path(query_images_path)
    sfm_dir = Path(sfm_model_path)
    features = Path(ref_features_fn)
    outputs = Path(output_path)

    # create output directory if it does not exist
    outputs.mkdir(parents=True, exist_ok=True)
    result_outputs = Path(output_path / Path("results"))
    if debug:
        result_outputs.mkdir(parents=True, exist_ok=True)

    # setup paths
    matches_fn = outputs / "matches.h5"
    loc_pairs_fn = outputs / "pairs_loc.txt"
    error_poses_fn = outputs / "pose_errors.txt"
    estimate_poses_fn = outputs / "estimate_poses.txt"

    # prepare ref and query image paths
    references_fn, references_fullp_fn = get_image_filenames(images)
    queries_fn, queries_fullp_fn = get_image_filenames(query_images)

    # parse cameras and poses
    cameras = get_cameras(poses_fn)
    poses = parse_poses_file(poses_fn, Twc=False, ext=".png")
    check_buffers(queries_fn, cameras)
    check_buffers(queries_fn, poses)

    # extract features for the query image
    extract_features.main(
        feature_conf,
        query_images,
        image_list=queries_fn,
        feature_path=features,
        overwrite=False,
    )

    # generate pairings and match features, skip if file already exists
    pairs_from_exhaustive.main(
        loc_pairs_fn, image_list=queries_fn, ref_list=references_fullp_fn
    )
    match_features.main(
        matcher_conf,
        loc_pairs_fn,
        features=features,
        matches=matches_fn,
        overwrite=False,
    )

    # read 3D model
    model = pycolmap.Reconstruction(sfm_dir)

    # localize query image
    ref_ids = [model.find_image_with_name(images / r).image_id for r in references_fn]
    localizer = QueryLocalizer(model, {"estimation": {"ransac": {"max_error": 12}}})

    fig = None
    if debug:
        fig3d = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig3d, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
    

    # write errors  and estimated poses to files
    error_pose_file = open(error_poses_fn, "w")
    estimate_pose_file = open(estimate_poses_fn, "w")
    # write header for estimated poses file: id, qw, qx, qy, qz, tx, ty, tz
    estimate_pose_file.write("# id qw qx qy qz tx ty tz\n")

    for query in queries_fn:
        cam_params = cameras[query]
        # localize and get pose

        ret, log = pose_from_cluster(
            localizer, query, cam_params, ref_ids, features, matches_fn
        )

        if ret is None:
            print(f"{query} - not correspondence found localization failed.")
            error_pose_file.write(str(query) + " nan nan \n")
            continue

        # print inliers count
        print(
            f'{query} - found {ret["num_inliers"]}/{len(ret["inlier_mask"])} inlier correspondences.'
        )

        # get localizated pose
        cam_from_world = ret["cam_from_world"]

        # make a fk 4x4, making the inversion here for later matrix product
        cam_in_world_estimate = np.vstack(
            [cam_from_world.inverse().matrix(), np.array([0, 0, 0, 1])]
        )

        qvec, tvec = pose_to_colmap_qt(np.linalg.inv(cam_in_world_estimate))
        estimate_pose_file.write(
            f"{query.strip('.png')} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]}\n"
        )

        # get GT world in camera
        Tcw = poses[query]


        def get_angle(R):
            cos_theta = (np.trace(R) - 1) / 2
            return np.arccos(cos_theta)

        T_diff = Tcw @ cam_in_world_estimate
        R_diff = T_diff[0:3, 0:3]

        angle_error_rad = get_angle(R_diff)
        t_diff = np.linalg.norm(T_diff[0:3, 3])

        # query img name, t_diff [m], r_diff [deg]
        error_pose_file.write(
            str(query).strip(".png")
            + " "
            + str(t_diff)
            + " "
            + str(np.rad2deg(angle_error_rad))
            + "\n"
        )

        # TODO if num inliers is lower than a threshold loc fail
        if debug:
            visualization.visualize_loc_from_log(query_images, query, log, model)
            fig = plt.gcf()  # ax = fig.axes
            output_file = result_outputs / f"{query}_results.png"
            fig.savefig(output_file, bbox_inches="tight")

    error_pose_file.close()
    matches_fn.unlink()
    loc_pairs_fn.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Localize query images using an SfM model and his features."
    )
    parser.add_argument(
        "--ref_images_path",
        type=str,
        help="path to the folder containing reference images",
        required=True,
    )
    parser.add_argument(
        "--query_images_path",
        type=str,
        help="path to the folder containing query images",
        required=True,
    )
    parser.add_argument(
        "--sfm_model_path",
        type=str,
        help="path to the folder containing the SFM model (COLMAP style)",
        required=True,
    )
    parser.add_argument(
        "--ref_features_fn",
        type=str,
        help="path to the file containing reference features",
        required=True,
    )
    # this is not required because camera model is fixed for challenge
    # parser.add_argument(
    #     "--cameras_fn", type=str,
    #     help="path to the file containing COLMAP camera intrinsics",
    #     required=True
    # )
    parser.add_argument(
        "--poses_fn",
        type=str,
        help="path to the file containing GT camera poses COLMAP style",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to the folder for saving outputs",
        required=True,
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    import shutil

    input_path = Path(args.ref_features_fn)
    if not input_path.is_file():
        raise FileNotFoundError(f"the file {args.ref_features_fn} does not exist.")

    # determine the new file's path
    feature_fn = Path(args.output_path) / "localization_features.h5"

    # copy the file
    shutil.copy(input_path, feature_fn)
    print(f"file copied from {input_path} to {feature_fn}")

    main(
        args.ref_images_path,
        args.query_images_path,
        args.sfm_model_path,
        feature_fn,
        # args.cameras_fn,
        args.poses_fn,
        args.output_path,
        args.debug,
    )

    feature_fn.unlink()
    print(f"file {feature_fn} removed")
