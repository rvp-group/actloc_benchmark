"""
ActLoc Data Parser

A utility for parsing training/test data from the ActLoc dataset.
Handles loading SfM models, waypoints, and localization results.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from utils.io import *
except ImportError:
    logging.error("Failed to import io from utils. Please ensure utils/io.py exists.")
    raise


class ActLocDataParser:
    """
    Parser for ActLoc dataset structure.

    Expected folder structure:
    root_dir/
    ├── raw_images/
    │   └── <scene_name>/
    │       ├── scene_reconstruction/
    │       ├── waypoint_1/
    │       ├── waypoint_2/
    │       ├── <scene_name>.glb
    │       └── sampled_viewpoints.txt
    └── sfm_and_localization_results/
        └── <scene_name>/
            ├── scene_reconstruction/
            ├── waypoint_1/
            └── waypoint_2/
    """

    def __init__(self, root_dir: str):
        """
        Initialize the data parser.

        Args:
            root_dir: Root directory containing raw_images and sfm_and_localization_results
        """
        self.root_dir = Path(root_dir)
        self.raw_images_dir = self.root_dir / "raw_images"
        self.sfm_results_dir = self.root_dir / "sfm_and_localization_results"

        # Validate directory structure
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        if not self.raw_images_dir.exists():
            raise FileNotFoundError(
                f"Raw images directory not found: {self.raw_images_dir}"
            )
        if not self.sfm_results_dir.exists():
            raise FileNotFoundError(
                f"SfM results directory not found: {self.sfm_results_dir}"
            )

        logging.info(f"ActLocDataParser initialized with root: {self.root_dir}")

    def list_scenes(self) -> List[str]:
        """
        List all available scenes in the dataset.

        Returns:
            List of scene names
        """
        scenes = []
        for item in self.raw_images_dir.iterdir():
            if item.is_dir() and (self.sfm_results_dir / item.name).exists():
                scenes.append(item.name)

        scenes.sort()
        logging.info(f"Found {len(scenes)} scenes: {scenes}")
        return scenes

    def get_mesh_file(self, scene_name: str) -> str:
        """
        Get the path to the 3D mesh file for a scene.
        Scene folder name format: 00005-yPKGKBCyYx8
        Mesh file name format: yPKGKBCyYx8.glb (without scene ID prefix)

        Args:
            scene_name: Name of the scene (e.g., '00005-yPKGKBCyYx8')

        Returns:
            Path to the mesh file
        """
        # Extract mesh name from scene name (remove scene ID prefix)
        if "-" in scene_name:
            mesh_name = scene_name.split("-", 1)[1]  # Get part after first dash
        else:
            mesh_name = scene_name  # Fallback if no dash found

        mesh_file = self.raw_images_dir / scene_name / f"{mesh_name}.glb"

        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        return str(mesh_file)

    def count_waypoints(self, scene_name: str) -> int:
        """
        Count the number of waypoints in a scene.

        Args:
            scene_name: Name of the scene

        Returns:
            Number of waypoints
        """
        waypoint_dirs = []
        scene_results_dir = self.sfm_results_dir / scene_name

        for item in scene_results_dir.iterdir():
            if item.is_dir() and item.name.startswith("waypoint_"):
                waypoint_dirs.append(item.name)

        return len(waypoint_dirs)

    def load_waypoint_pose_errors(
        self, scene_name: str, waypoint_idx: int
    ) -> Optional[Dict]:
        """
        Load pose errors for a specific waypoint.

        Args:
            scene_name: Name of the scene
            waypoint_idx: Waypoint index (1-based)

        Returns:
            Dictionary with pose error information, or None if not found
        """
        pose_errors_file = (
            self.sfm_results_dir
            / scene_name
            / f"waypoint_{waypoint_idx}"
            / "scene_reconstruction"
            / "pose_errors.txt"
        )

        if not pose_errors_file.exists():
            logging.warning(f"Pose errors file not found: {pose_errors_file}")
            return None

        try:
            pose_errors = self._parse_pose_errors_file(str(pose_errors_file))
            logging.info(
                f"Loaded pose errors for {scene_name}/waypoint_{waypoint_idx}: {len(pose_errors)} entries"
            )
            return pose_errors
        except Exception as e:
            logging.error(f"Failed to parse pose errors file {pose_errors_file}: {e}")
            return None

    def load_waypoint_ground_truth_poses(
        self, scene_name: str, waypoint_idx: int
    ) -> Optional[Dict]:
        """
        Load ground truth poses for all viewpoints at a specific waypoint.

        Args:
            scene_name: Name of the scene
            waypoint_idx: Waypoint index (1-based)

        Returns:
            Dictionary with ground truth pose information, or None if not found
        """
        gt_poses_file = (
            self.raw_images_dir
            / scene_name
            / f"waypoint_{waypoint_idx}"
            / "img_name_to_colmap_Tcw.txt"
        )

        if not gt_poses_file.exists():
            logging.warning(f"Ground truth poses file not found: {gt_poses_file}")
            return None

        try:
            gt_poses = self._parse_poses_file(str(gt_poses_file))
            logging.info(
                f"Loaded ground truth poses for {scene_name}/waypoint_{waypoint_idx}: {len(gt_poses)} entries"
            )
            return gt_poses
        except Exception as e:
            logging.error(
                f"Failed to parse ground truth poses file {gt_poses_file}: {e}"
            )
            return None

    def _parse_pose_errors_file(self, file_path: str) -> Dict:
        """Parse pose_errors.txt file."""
        pose_errors = {}

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 3:
                image_name = parts[0]
                trans_error = float(parts[1])
                rot_error = float(parts[2])
                pose_errors[image_name] = {
                    "translation_error": trans_error,
                    "rotation_error": rot_error,
                }

        return pose_errors

    def load_waypoint_localization_results(
        self, scene_name: str, waypoint_idx: int
    ) -> Optional[Dict]:
        """
        Load localization results for a specific waypoint.

        Args:
            scene_name: Name of the scene
            waypoint_idx: Waypoint index (1-based)

        Returns:
            Dictionary with localization results, or None if not found
        """
        results_file = (
            self.sfm_results_dir
            / scene_name
            / f"waypoint_{waypoint_idx}"
            / "scene_reconstruction"
            / "results.txt"
        )

        if not results_file.exists():
            logging.warning(f"Results file not found: {results_file}")
            return None

        try:
            results = self._parse_poses_file(str(results_file))
            logging.info(
                f"Loaded localization results for {scene_name}/waypoint_{waypoint_idx}: {len(results)} entries"
            )
            return results
        except Exception as e:
            logging.error(f"Failed to parse results file {results_file}: {e}")
            return None

    def _parse_poses_file(self, file_path: str) -> Dict:
        """Parse pose files (both results.txt and img_name_to_colmap_Tcw.txt have same format)."""
        poses = {}

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 8:  # Format: image_name qw qx qy qz tx ty tz
                image_name = parts[0]
                # Quaternion (w, x, y, z) and translation (x, y, z)
                qw, qx, qy, qz = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

                poses[image_name] = {
                    "quaternion": [qw, qx, qy, qz],  # [w, x, y, z] format
                    "translation": [tx, ty, tz],
                }

        return poses

    def load_scene_data(self, scene_name: str) -> Dict[str, Any]:
        """
        Load all data for a scene including localization data.

        Args:
            scene_name: Name of the scene

        Returns:
            Dictionary containing all scene data
        """
        logging.info(f"Loading complete data for scene: {scene_name}")

        # Load basic data

        waypoints_file = self.raw_images_dir / scene_name / "sampled_viewpoints.txt"
        waypoints = load_waypoints(waypoints_file)
        sfm_dir = (
            self.sfm_results_dir / scene_name / "scene_reconstruction" / "reference_sfm"
        )
        cameras, images, points3D = load_sfm_model(str(sfm_dir))
        mesh_file = self.get_mesh_file(scene_name)
        num_waypoints = self.count_waypoints(scene_name)

        scene_data = {
            "scene_name": scene_name,
            "waypoints": waypoints,
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "mesh_file": mesh_file,
            "num_waypoints": num_waypoints,
            "waypoint_data": {},
        }

        # Load waypoint-specific data
        for waypoint_idx in range(1, num_waypoints + 1):
            waypoint_data = {}

            # Load pose errors
            pose_errors = self.load_waypoint_pose_errors(scene_name, waypoint_idx)
            if pose_errors is not None:
                waypoint_data["pose_errors"] = pose_errors

            # Load localization results (estimated poses)
            loc_results = self.load_waypoint_localization_results(
                scene_name, waypoint_idx
            )
            if loc_results is not None:
                waypoint_data["estimated_poses"] = loc_results

            # Load ground truth poses
            gt_poses = self.load_waypoint_ground_truth_poses(scene_name, waypoint_idx)
            if gt_poses is not None:
                waypoint_data["ground_truth_poses"] = gt_poses

            scene_data["waypoint_data"][waypoint_idx] = waypoint_data

        logging.info(f"Loaded complete data for scene {scene_name}")
        return scene_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ActLoc Data Parser Example")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="full_actloc_data/test_data",
        help="Root directory containing training/test data",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="00005-yPKGKBCyYx8",
        help="Specific scene to load (if not specified, lists all scenes)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        # Initialize parser
        data_parser = ActLocDataParser(args.root_dir)

        if args.scene_name is None:
            # List all scenes
            scenes = data_parser.list_scenes()
            print(f"\nFound {len(scenes)} scenes:")
            for i, scene in enumerate(scenes, 1):
                waypoint_count = data_parser.count_waypoints(scene)
                print(f"  {i:2d}. {scene} ({waypoint_count} waypoints)")
        else:
            # Load specific scene
            scene_data = data_parser.load_scene_data(args.scene_name)

            print(f"\n=== Scene: {args.scene_name} ===")
            print(f"Waypoints: {scene_data['waypoints'].shape}")
            print(
                f"SfM - Images: {len(scene_data['images'])}, Points: {len(scene_data['points3D'])}"
            )
            print(f"Mesh file: {scene_data['mesh_file']}")
            print(f"Number of waypoints: {scene_data['num_waypoints']}")

            print("\nWaypoint localization data:")
            for wp_idx, wp_data in scene_data["waypoint_data"].items():
                pose_errors = wp_data.get("pose_errors", {})
                estimated_poses = wp_data.get("estimated_poses", {})
                gt_poses = wp_data.get("ground_truth_poses", {})
                print(
                    f"  Waypoint {wp_idx}: {len(pose_errors)} pose errors, {len(estimated_poses)} estimated poses, {len(gt_poses)} GT poses"
                )

            # Show first few waypoints
            print("\nFirst few waypoints:")
            for i, waypoint in enumerate(scene_data["waypoints"][:5]):
                print(
                    f"  Waypoint {i+1}: [{waypoint[0]:8.3f}, {waypoint[1]:8.3f}, {waypoint[2]:8.3f}]"
                )

            if len(scene_data["waypoints"]) > 5:
                print(f"  ... and {len(scene_data['waypoints']) - 5} more")

            # Show sample data from first waypoint if available
            if scene_data["waypoint_data"]:
                wp1_data = scene_data["waypoint_data"][1]
                if "pose_errors" in wp1_data and wp1_data["pose_errors"]:
                    sample_image = list(wp1_data["pose_errors"].keys())[0]
                    sample_error = wp1_data["pose_errors"][sample_image]
                    print(f"\nSample pose error for {sample_image}:")
                    print(
                        f"  Translation error: {sample_error['translation_error']:.6f}m"
                    )
                    print(f"  Rotation error: {sample_error['rotation_error']:.6f}°")

                if "ground_truth_poses" in wp1_data and wp1_data["ground_truth_poses"]:
                    sample_image = list(wp1_data["ground_truth_poses"].keys())[0]
                    sample_gt = wp1_data["ground_truth_poses"][sample_image]
                    print(f"\nSample ground truth pose for {sample_image}:")
                    print(f"  Quaternion [w,x,y,z]: {sample_gt['quaternion']}")
                    print(f"  Translation [x,y,z]: {sample_gt['translation']}")

    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
