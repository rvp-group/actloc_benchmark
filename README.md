# Active Localization Benchmark

Active localization is the task of determining the most informative viewpoints to improve a robot’s pose estimation within a known map. Traditionally, a robot navigating an environment may rely on passive localization—simply pointing its camera forward while moving. However, this ignores the fact that not all viewpoints are equally informative. Consider a scenario where the robot faces a featureless white wall: looking straight ahead in this case offers little to no localization cues.

<!-- <p align="center">
  <img src="assets/forward1.png" width="30%" alt="passloc1"/>
  <img src="assets/forward2.png" width="30%" alt="passloc2"/>
  <img src="assets/forward3.png" width="30%" alt="passloc3"/>
</p> -->

Active localization instead aims to reason about the environment and proactively select viewpoints that maximize perceptual information, improving robustness and accuracy, especially in ambiguous or textureless areas. Heuristics in where to look can be achieved through optimiality criteria and Fisher information matrics, visibility, distribution of landmarks or deep learned techniques. 

### Important submission information
Before going through this repo we assumed you have read important information about submission [activep-ws.github.io/challenge](https://activep-ws.github.io/challenge.html)


<!-- <p align="center">
  <img src="images/placeholder.png" width="30%" alt="Image 1"/>
</p> -->

## How to Actively Localize?
The workflow is simple, you can start from any inputs SfM reconstruction, mesh, semantics image, the only **mandatory** thing are the waypoints, 3D locations (x, y, z) from which you want to provide the "best" viewing angles:

```
input: any <SfM model, mesh, semantics, etc..> + 3D waypoints (mandatory)
         ↓
    inference.py
         ↓
    best viewing angles (we assume for each 3D location you produce an orientation in form of quaternion)
```

This repository provides an end-to-end pipeline for predicting optimal camera viewing directions at given waypoints in 3D scenes and capturing images at those predicted viewpoints.

## Environment Setup
Once you clone the repo, make sure you clone its submodules. The localization accuracy calculation is based on [hloc](https://github.com/cvg/Hierarchical-Localization), hence requires some features and matchers locally. Either you clone with flag `--recursive` or you clone and update the submodules this way:
```bash
git submodule update --init --recursive
```
Create conda environment:
```bash
conda create -n actloc_benchmark  python=3.11 && conda activate actloc_benchmark
```
and install
```bash
pip install -r requirements.txt && pip install -e .
```

### Download some sample data

Download sample data, this consists on a sample mesh `.glb` and an SfM model with cameras, points and images:
```bash
chmod +x download_sample_data.sh && ./download_sample_data.sh
```

## Overview
The pipeline consists of four main components:

1. **`inference.py`** - Predicts best viewing angles for each waypoint (this is where you should put your hands for the challenge),
here you need to output the orientations. 
2. **`capture_images_at_best_viewing_directions.py`** - Captures images at the predicted best orientations, this is required to evaluate the localization accuracy.
3. **`match_and_localize.py`** - Given poses and their corresponding images from step (1) and (2) you want to match and localize against the SfM model. For this purpose we employ [hloc](https://github.com/cvg/Hierarchical-Localization). For each image localized you will have an error.
4. **`evaluate.py`** - Given the errors from localization, this outputs the localization score according to [learning-where-to-look](https://link.springer.com/chapter/10.1007/978-3-031-73016-0_12).

We now explain how each module work and what are I/O through a practical example.

### To work with a specific scene, navigate to its data folder:
```bash
cd actloc_benchmark/example_data/00005-yPKGKBCyYx8/
```


### `inference.py`
Predict the best viewing angles for your waypoints (here is where you should put your hands) for now the best angles heuristics is simply based on maximizing the visibility of 3d landmarks. The scripts output full estimate pose (waypoint + orientation) in COLMAP style:
```bash
python ../../inference.py \
    --waypoints-file sampled_waypoints.txt \
    --sfm-dir scene_reconstruction \
    --output-estimate estimate/selected_gt_poses.txt
```

### `capture_images_at_best_viewing_directions.py`
Capture images corresponding to previously estimated poses:
```bash
python ../../capture_images_at_best_viewing_directions.py \
    --pose-file ../00005-yPKGKBCyYx8/estimate/selected_gt_poses.txt \
    --mesh-file yPKGKBCyYx8.glb \
    --output-folder estimate/images
```

### `match_and_localize.py`
```bash
python ../../match_and_localize.py \
    --sfm_model_path scene_reconstruction \
    --ref_images_path scene_reconstruction/images \
    --ref_features_fn scene_reconstruction/sfm_features.h5 \
    --query_images_path estimate/images \
    --poses_fn estimate/selected_gt_poses.txt \
    --output_path estimate
```

### `vis.py`
If you want to debug we provide a visualization utility, to visualize different poses. 

> [!TIP]
> Make sure you don't have roll - rotation around the z-optical axis in you "best viewpoints" images.

### TODO Boyang

### `evaluate_loc.py`
```bash
python ../../evaluate_loc.py --error-file estimate/pose_errors.txt
```
Note that evaluation for single viewpoint localization is based on accuracy intervals, if you want to calculate the accuracy among multiple scenes is enough that you concatenate each `error-file` into a single file and input this to `evaluate_loc.py`. Do not average results, it is not the correct way!

### `evaluate_plan.py`
### TODO Boyang

## Data provided

### File formats and explanation
Here we explain only file formats and what is the data we provide you as sample, if you are familiar with COLMAP you can skip most of this part:

- **SfM Reconstruction**: COLMAP reconstruction folder containing:
  - `cameras.bin/txt` - Camera intrinsics
  - `images.bin/txt` - Camera poses and image/features information
  - `points3D.bin/txt` - 3D point cloud with colors
- **Waypoints File**: Text file with 3D coordinates (N×3 format)
  ```
  x1 y1 z1
  x2 y2 z2
  ...
  ```
- **Mesh File**: 3D scene mesh (`.glb`, etc.)

### Full Dataset 
We provide you with a sample dataset including 90 meshes and their SfM model that you can use for training or test the robustness of your approach. You can download data from [here](https://drive.google.com/file/d/1OyFqkwyBWCA7iDw-GLIXK3xRWPjnATYC/view?usp=drive_link). This contains more scene folder similar to sample data:
#### TODO
Explain format of full data


### Full Dataset with Viewpoints
In addition we provide some data that you could potentially employ for training. This contains already for each sampled waypoint, is pontetial camera orientations and the captured images. So basically the full preprocess is ready for you. Bear in mind that this has been collected at the following orientation resolution:
- **elevation-axis**: 6 intervals covering [-60°, +40°] with 20° steps
- **azimuthal-axis**: 18 intervals covering [-180°, +160°] with 20° steps

You can download it from [here](https://drive.google.com/drive/folders/1vsMV2CI144ihui4oJHrykwCkq0xKWJrJ?usp=sharing) and it is organized as follows:
```
training_data
├── raw_images
│   ├── <scene_1>
│   │   ├── scene_reconstruction
│   │   │   ├── images
│   │   │   ├── img_nm_to_colmap_cam.txt
│   │   │   └── img_name_to_colmap_Tcw.txt
│   │   ├── waypoint_1
│   │   │   ├── images
│   │   │   ├── img_nm_to_colmap_cam.txt
│   │   │   └── img_name_to_colmap_Tcw.txt
│   │   ├── waypoint_2
│   │   │   └── …
│   │   ├── <scene_1>.glb
│   │   ├── sampled_viewpoints.txt
│   │   └── ...
│   ├── <scene_2>
│   │   └── ...
│   └── ...
└── sfm_and_localization_results
    ├── <scene_1>
    │   ├── scene_reconstruction
    │   │   ├── cameras.bin
    │   │   ├── images.bin
    │   │   └── points3D.bin
    │   ├── waypoint_1
    │   │   ├── scene_reconstruction
    │   │   │   ├── results.txt
    │   │   │   └── pose_errors.txt
    │   ├── waypoint_2
    │   │   └── …
    │   └── ...
    ├── <scene_2>
    │   └── ...
    └── ...
```